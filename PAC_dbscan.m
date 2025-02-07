% -----------------------------------------------------------------------------------------------------------
% PAC analysis using DBSCAN
% Written by Srijita Das
% Modified on July 30, 2024

clear
%% Step 1: Define input data
% Addpath
addpath("U:\shared\users\sdas\brainstorm_db\PAC_current\tensor_toolbox-v3.6");

% Load file containing scout names
load("Destrieux_row_names.mat")

% Define the base directory where your data is stored
base_dir = 'U:\shared\users\sdas\meg-UNMC_results\';

% Define tasks and sides
tasks = {'SEF','SEF','spont'};
datafile_name = {'SEFul','SEFur','spont'};

% Get the list of patients in base directory
patients_all = dir(base_dir);
patient_names = {patients_all([patients_all.isdir]).name};
patient_names = patient_names(~ismember(patient_names, {'.','..'}));
patient_names = patient_names(1:137);                                     % select the range of patients you want to include

%% Step 2: Load PAC data for each task

warning('off', 'MATLAB:structOnObject');

% Initialize variables
all_patients_data = zeros(148,8,81,30,3);                                % in the form of scouts x LF x HF x patients x task
patients_with_pac = {};

% Loop through patients
k = 1;                                                                   % to account for the number of patients
for i = 1:length(patient_names)
    subjectID = patient_names{i};

    % Loop through tasks
    for j = 1:length(tasks)
        data_path = fullfile(base_dir,subjectID,'PAC',tasks{j},[datafile_name{j},'.mat']);

        % Check if PAC_data exists
        rng('default')
        if exist(data_path,'file')
            m = matfile( data_path );

            if( isfield( struct(m), 'sPAC') )
                fprintf('%s\n', data_path );

                % Load Direct PAC data
                sPAC = m.sPAC;                                  %contains info on DirectPAC,nestingfreq,nestedfreq,LF,HF
                direct_PAC = sPAC.DirectPAC;
                
                direct_PAC_reshaped = reshape(direct_PAC,[],8,81);

                % Concatenate data across all patients
                all_patients_data(:,:,:,k,j) = direct_PAC_reshaped;
                if( j == 3)
                    patients_with_pac{end+1} = subjectID;
                    k = k+1;
                end
            end
        end
    end
end

%% Step 3: Decompose tensor for each rank and perform clustering
direct_PAC_5D_tensor = tensor(all_patients_data);
dec_product = cell(2,3);
% Decomposition of 4-D tensor for each task
for rank = 2
    for t = 1:3
    PAC_data = direct_PAC_5D_tensor(:,:,:,:,t);
    dec_product{rank,t} = cp_als(PAC_data,rank,'tol',1e-6,'maxiters',1000);
    info = viz(dec_product{rank,t}, 'PlotCommands', {'scatter','bar','bar','bar'}, ...
        'ModeTitles', {'PAC values','Low freq','High freq','Patient'}, ...
        'Figure', [], 'BaseFontSize', 18);
    hold on;
    % Extract the PAC values
    PAC_values = dec_product{rank,t}.U{1};
    % normalize PAC values across all dimensions
    norm_PAC_values = zscore(PAC_values);

    % Perform DBSCAN clustering
    eps_range = 0.5:0.1:2;                                          % Adjust the epsilon range as needed
    minPts_range = 2:10;                                            % Adjust the MinPts range as needed
    best_eps = [];
    best_minPts = [];
    best_silhouette = -inf;

    for eps = eps_range
        for minPts = minPts_range
            [cluster_idx, ~] = dbscan(norm_PAC_values, eps, minPts);
            if numel(unique(cluster_idx)) > 1
                silhouette_val = silhouette(norm_PAC_values, cluster_idx);
                avg_silhouette = mean(silhouette_val);
                if avg_silhouette > best_silhouette
                    best_silhouette = avg_silhouette;
                    best_eps = eps;
                    best_minPts = minPts;
                end
            end
        end
    end
    fprintf('Best epsilon: %.2f, Best MinPts: %d\n', best_eps, best_minPts);

    [cluster_idx, ~] = dbscan(norm_PAC_values,best_eps,best_minPts);                % epsilon, minpts

    % Plot the clustering results
    figure;
    gscatter(norm_PAC_values(:,1), norm_PAC_values(:,2), cluster_idx, [0.4660 0.6740 0.1880;0.3010 0.7450 0.9330], ...
        ['h','v'],[11 15],['','filled']);
    xlabel('PC1', 'FontWeight','bold','FontSize',20);
    ylabel('PC2','FontWeight','bold','FontSize',20);
    
    fontsize(19, "points");
    legend('','Regions of interest');

    %title(sprintf('Clustering Results for Task %s Rank %d (Eps=%.2f, MinPts=%d)', datafile_name{t}, rank, best_eps, best_minPts));
    hold on;
    end
end

%% Step 4: Calculate probability using binomial distribution
% Use this section after visual inspection of the number of observed
% regions for each task
% n = 148; % total no. of regions
% k = 6;   % number of expected regions
% r = 5;  % number of active regions identified
% e = 5;   % number of expected regions among the active ones
% 
% % probability of an individual region being an expected region
% p = k / n;
% 
% % Calculate the cumulative probability using binocdf
% prob = 1 - binocdf(e, r, p);
% 
% % Display the result
% fprintf('The probability of getting at least %d expected regions out of %d active regions is: %.2e\n', e, r, prob);


