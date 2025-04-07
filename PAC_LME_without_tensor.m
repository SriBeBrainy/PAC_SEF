% -----------------------------------------------------------------------------------------------------------
% PAC analysis using Mixed-effects model with permutation test
% This method uses the maxPAC, no tensor decomposition
% Two LMEs for left and right, with permutation testing
% Written by Srijita Das on July 31, 2024
% Modified on August 16, 2024

%%
clear

%% Step 1: Define input data

% Load file containing scout names
load("Destrieux_row_names.mat");

% Define the base directory where your data is stored
base_dir = 'U:\shared\users\sdas\meg-UNMC_results\';

% Define tasks and sides
tasks = {'SEF', 'SEF', 'spont'};
datafile_name = {'SEFul', 'SEFur', 'spont'};

% Get the list of patients in base directory
patients_all = dir(base_dir);
patient_names = {patients_all([patients_all.isdir]).name};
patient_names = patient_names(~ismember(patient_names, {'.', '..'}));
patient_names = patient_names(1:137); % select the range of patients you want to include

% Define expected brain regions
exp_indices_UL = [52, 56, 58, 90, 134, 138];
exp_indices_UR = [51, 55, 57, 89, 133, 137];

%% Step 2: Create table

% Initialize variables
maxPAC_values_left = [];                          % for both SEF-UL and spont
maxPAC_values_right = [];                         % for both SEF-UR and spont
factors_left = [];
factors_right = [];
patient_IDs_left = [];
patient_IDs_right = [];

patient_index = 1; % Counter for patient IDs

% Loop through each patient
for p = 1:length(patient_names)
    subjectID = patient_names{p};

    % Initialize flags to check if PAC data exists for each task
    has_pac_SEFul = false;
    has_pac_SEFur = false;
    has_pac_spont = false;

    % Initialize variables for storing maxPAC values
    maxPAC_SEFul = [];
    maxPAC_SEFur = [];
    maxPAC_spont = [];

    % Loop through tasks
    for j = 1:length(tasks)
        data_path = fullfile(base_dir, subjectID, 'PAC', tasks{j}, [datafile_name{j}, '.mat']);
        
        % Check if PAC_data exists
        warning('off', 'MATLAB:structOnObject');
        rng('default')
        if exist(data_path, 'file')
            m = matfile(data_path);
            if isfield(struct(m), 'sPAC')
                switch tasks{j}
                    case 'SEF'
                        if strcmp(datafile_name{j}, 'SEFul')
                            has_pac_SEFul = true;
                            maxPAC_SEFul = m.TF; 
                            fprintf('%s\n', data_path);
                        else
                            has_pac_SEFur = true;
                            maxPAC_SEFur = m.TF;
                            fprintf('%s\n', data_path );
                        end
                    case 'spont'
                        has_pac_spont = true;
                        maxPAC_spont = m.TF;
                        fprintf('%s\n', data_path );
                end
            end
        end
    end
    
    % Append the data for each patient
    if has_pac_spont
        maxPAC_values_left = [maxPAC_values_left; maxPAC_spont(:)];
        factors_left = [factors_left; repmat(1, 148, 1)];
        patient_IDs_left = [patient_IDs_left; repmat(patient_index, 148, 1)];
        
        maxPAC_values_right = [maxPAC_values_right; maxPAC_spont(:)];
        factors_right = [factors_right; repmat(1, 148, 1)];
        patient_IDs_right = [patient_IDs_right; repmat(patient_index, 148, 1)];
    end
    
    if has_pac_SEFul
        for k = 1:148
            if ismember(k, exp_indices_UL)
                factors_left = [factors_left; 3];
            else
                factors_left = [factors_left; 2];
            end
        end
        maxPAC_values_left = [maxPAC_values_left; maxPAC_SEFul(:)];
        patient_IDs_left = [patient_IDs_left; repmat(patient_index, 148, 1)];
    end
    
    if has_pac_SEFur
        for k = 1:148
            if ismember(k, exp_indices_UR)
                factors_right = [factors_right; 3];
            else
                factors_right = [factors_right; 2];
            end
        end
        maxPAC_values_right = [maxPAC_values_right; maxPAC_SEFur(:)];
        patient_IDs_right = [patient_IDs_right; repmat(patient_index, 148, 1)];
    end

    % Increment patient index if PAC data was found for this patient
    if has_pac_spont || has_pac_SEFul || has_pac_SEFur
        patient_index = patient_index + 1;
    end
end

%% Step 3: Create tables for left and right data
data_left = table(maxPAC_values_left, categorical(factors_left), categorical(patient_IDs_left), ...
    'VariableNames', {'MaxPAC', 'Factor', 'PatientID'});
data_right = table(maxPAC_values_right, categorical(factors_right), categorical(patient_IDs_right), ...
    'VariableNames', {'MaxPAC', 'Factor', 'PatientID'});
% statset('linearmixedmodel')
% options = statset('TolFun',1e-6,'TolX',1e-12,'MaxIter',10000);

%% Step 4: Fit linear mixed effects models
% fixed-effect: task (factor), random-effect: patients, response: maxPAC
lme_left = fitlme(data_left, 'MaxPAC ~ Factor + (1|PatientID)');
lme_right = fitlme(data_right, 'MaxPAC ~ Factor + (1|PatientID)');

% Display the models
disp(lme_left);
disp(lme_right);

% Original estimates
original_estimate_factor2_left = lme_left.Coefficients.Estimate(2);
original_estimate_factor3_left = lme_left.Coefficients.Estimate(3);
original_estimate_factor2_right = lme_right.Coefficients.Estimate(2);
original_estimate_factor3_right = lme_right.Coefficients.Estimate(3);

%% Step 5: Permutation testing
num_permutations = 10000;
perm_estimate_left = zeros(num_permutations, 2);  % Store p-values for Factor 2 and Factor 3
perm_estimate_right = zeros(num_permutations, 2);

rng('default');
for i = 1:num_permutations
    % Permute maxPAC values for each patient keeping Factor and PatientID the same
    permuted_maxPAC_left = zeros(size(maxPAC_values_left));
    for ii=1:patient_index-1
        I = patient_IDs_left == ii;
        temp_left = maxPAC_values_left(I);
        permuted_maxPAC_left(I) = temp_left(randperm(length(temp_left)));
    end
    permuted_maxPAC_right = zeros(size(maxPAC_values_right));
    for jj=1:patient_index-1
        J = patient_IDs_right == jj;
        temp_right = maxPAC_values_right(J);
        permuted_maxPAC_right(J) = temp_right(randperm(length(temp_right)));
    end
    
    % Create new tables with permuted maxPAC values
    permuted_data_left = table(permuted_maxPAC_left, categorical(factors_left), categorical(patient_IDs_left), ...
        'VariableNames', {'Perm_MaxPAC', 'Factor', 'PatientID'});
    permuted_data_right = table(permuted_maxPAC_right, categorical(factors_right), categorical(patient_IDs_right), ...
        'VariableNames', {'Perm_MaxPAC', 'Factor', 'PatientID'});
    
    % Fit mixed-effects models to permuted data
    permuted_lme_left = fitlme(permuted_data_left, 'Perm_MaxPAC ~ Factor + (1|PatientID)');
    permuted_lme_right = fitlme(permuted_data_right, 'Perm_MaxPAC ~ Factor + (1|PatientID)');

    % Extract estimates for Factor 2 and Factor 3
    perm_estimate_left(i, :) = permuted_lme_left.Coefficients.Estimate(2:3)';
    perm_estimate_right(i, :) = permuted_lme_right.Coefficients.Estimate(2:3)';
end

disp(permuted_lme_left)
disp(permuted_lme_right)

%% Step 6: Analyze Permutation Results
% left
p_value_factor2_left = sum(abs(perm_estimate_left(:, 1)) >= abs(original_estimate_factor2_left)) / num_permutations;
p_value_factor3_left = sum(abs(perm_estimate_left(:, 2)) >= abs(original_estimate_factor3_left)) / num_permutations;

% right
p_value_factor2_right = sum(abs(perm_estimate_right(:, 1)) >= abs(original_estimate_factor2_right)) / num_permutations;
p_value_factor3_right = sum(abs(perm_estimate_right(:, 2)) >= abs(original_estimate_factor3_right)) / num_permutations;

%% Plots
% Relationship btw fixed and response
figure;
plot(permuted_data_left.Factor,permuted_data_left.Perm_MaxPAC,'bo');
xlabel("Factors")
ylabel('Max PAC')

% Residuals vs fitted 
figure;
plotResiduals(lme_left,"fitted")
hold on

% error bar plot
% Original estimates and SE
orig_estimates_left = [original_estimate_factor2_left, original_estimate_factor3_left];
orig_estimates_right = [original_estimate_factor2_right, original_estimate_factor3_right];

% Standard errors 
se_left = [lme_left.Coefficients.SE(2), lme_left.Coefficients.SE(3)];
se_right = [lme_right.Coefficients.SE(2), lme_right.Coefficients.SE(3)];

% Confidence intervals
ci_left = 1.96 * se_left;
ci_right = 1.96 * se_right;

% Permuted estimates mean
perm_mean_left = mean(perm_estimate_left);
perm_mean_right = mean(perm_estimate_right);

%% Step 7: Generate Boxplot with Error Bars

% Combine permuted estimates and group them for plotting
perm_data = [perm_estimate_left; perm_estimate_right];
group = [repmat({'Left-Factor 2'}, num_permutations, 1); repmat({'Left-Factor 3'}, num_permutations, 1); 
         repmat({'Right-Factor 2'}, num_permutations, 1); repmat({'Right-Factor 3'}, num_permutations, 1)];

% Create the boxplot
figure;
boxplot(perm_data, group,"OutlierSize",8);
hold on;

% Add error bars for original estimates
x_positions = [1, 2, 3, 4]; % x-axis positions for Left-Factor 2, Left-Factor 3, Right-Factor 2, Right-Factor 3
errorbar(x_positions, [original_estimate_factor2_left, original_estimate_factor3_left, original_estimate_factor2_right, original_estimate_factor3_right], ...
    [ci_left, ci_right], 'ko', 'MarkerSize', 10, 'LineWidth', 1.5, 'CapSize', 10);

% Add dots for original estimates
plot(x_positions, [original_estimate_factor2_left, original_estimate_factor3_left, original_estimate_factor2_right, original_estimate_factor3_right], ...
    'co', 'MarkerSize', 12, 'MarkerFaceColor', 'c');

% Customize the plot
xlabel('Factors and Sides','FontSize', 16, 'FontWeight', 'bold');
ylabel('Estimates','FontSize', 16, 'FontWeight', 'bold');
%title('Comparison of original and permuted estimates for factor 2 and 3');
grid on;
legend('Error bar','Original estimates');

% Adjust y-axis limits for better visualization
ylim([min(min(perm_data)) - 0.02, max(max(perm_data)) + 0.08]);

% Show plot
hold off;

