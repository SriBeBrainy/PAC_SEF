% -----------------------------------------------------------------------------------------------------------
% PAC analysis using Mixed-effects model
% Written by Srijita Das
% Modified on July 30, 2024

%%
clear

%% Step 1: Define input data
% Addpath
addpath("C:\brainstorm_db\PAC\tensor_toolbox-v3.6");

% Load file containing scout names
load("Destrieux_row_names.mat");

% Define the base directory where your data is stored
base_dir = 'U:\shared\users\sdas\meg-UNMC_results\';

% Define tasks and sides
tasks = {'SEF','SEF','spont','spont'};
datafile_name = {'SEFul','SEFur','spont','spont'};

% Get the list of patients in base directory
patients_all = dir(base_dir);
patient_names = {patients_all([patients_all.isdir]).name};
patient_names = patient_names(~ismember(patient_names, {'.','..'}));

%% Step 2: Model

% Initialize variables before start of loop for the table
pt = 32; % number of patients with PAC
PAC_values = cell(pt,4);

patient_ID_table = repelem((1:pt)', 148, 1);
scout_number = repmat((1:148)', pt, 1);
exp_region_table = false(148*pt,1);    
PAC_values_table = zeros(148*pt,1);
%task = cell();

% Loop through each task
for j = 1:length(tasks)
    % Initialize variables for each task
    exp_indices_UL = [];
    exp_indices_UR = [];

    % Loop through each patient with PAC data
    pac_patients = 0;
    for p = 1:length(patient_names)
        subjectID = patient_names{p};
        has_pac = false; % Flag to check if the patient has PAC data

        data_path = fullfile(base_dir, subjectID, 'PAC', tasks{j}, [datafile_name{j}, '.mat']);

        % Check if PAC_data exists
        warning('off', 'MATLAB:structOnObject');
        rng('default')
        if exist(data_path, 'file')
            m = matfile(data_path);
            if isfield(struct(m), 'sPAC')
                has_pac = true;
                pac_patients = pac_patients + 1;
            end
        end

        if has_pac && pac_patients <= pt
            % Load Direct PAC data
            sPAC = m.sPAC;
            direct_PAC = sPAC.DirectPAC;

            direct_PAC_reshaped = reshape(direct_PAC, [], 14, 74); % converting into 3D matrix to remove time component

            % Step 3: Decompose tensor
            direct_PAC_3D_tensor = tensor(direct_PAC_reshaped);

            % Decomposition on 3-D tensor for each rank
            rank = 3;
            PAC = cp_als(direct_PAC_3D_tensor, rank, 'maxiters', 50);

            % Extract the PAC values only along 1st dimension
            PAC_values{pac_patients, j} = zscore(PAC.U{1}(:,1));

            % Arrange corresponding PAC values aligned with table
            start_idx = (pac_patients - 1) * 148 + 1;
            end_idx = pac_patients * 148;
            PAC_values_table(start_idx:end_idx) = PAC_values{pac_patients, j};

            % Create matrix for expected brain regions
            if j == 1||3
                exp_indices_UL = [52, 56, 58, 90, 134, 138];
            end
            %include spont(j=1&3), combine exp_reg is member
            if j == 2||4
                exp_indices_UR = [51, 55, 57, 89, 133, 137];
            end

            % Update exp_region_table based on the task
            if j == 1 && ~isempty(exp_indices_UL)
                exp_region_table(start_idx:end_idx) = ismember(1:148, exp_indices_UL)';
            elseif j == 3 && ~isempty(exp_indices_UL)
                exp_region_table(start_idx:end_idx) = ismember(1:148, exp_indices_UL)';
            elseif j == 2 && ~isempty(exp_indices_UR)
                exp_region_table(start_idx:end_idx) = ismember(1:148, exp_indices_UR)';
            elseif j == 4 && ~isempty(exp_indices_UR)
                exp_region_table(start_idx:end_idx) = ismember(1:148, exp_indices_UR)';
            end
        end
    end

    % Create tables for each task                %when j=3&4 pac_values same but exp_reg change
    if j ==1
        task_table_SEF_UL = table(patient_ID_table(1:pac_patients*148), exp_region_table(1:pac_patients*148), ...
            PAC_values_table(1:pac_patients*148), ...
            'VariableNames', {'Patient_ID', 'Expected_Region', 'PAC_Values'});

        lme_SEF_UL = fitlme(task_table_SEF_UL,'PAC_Values ~ Expected_Region + (1|Patient_ID)','FitMethod','ML');

    elseif j==2
        task_table_SEF_UR = table(patient_ID_table(1:pac_patients*148), exp_region_table(1:pac_patients*148), ...
            PAC_values_table(1:pac_patients*148), ...
            'VariableNames', {'Patient_ID', 'Expected_Region', 'PAC_Values'});

        lme_SEF_UR = fitlme(task_table_SEF_UR,'PAC_Values ~ Expected_Region + (1|Patient_ID)','FitMethod','ML');

    elseif j==3
        task_table_spont_1 = table(patient_ID_table(1:pac_patients*148), exp_region_table(1:pac_patients*148), ...
            PAC_values_table(1:pac_patients*148), ...
            'VariableNames', {'Patient_ID', 'Expected_Region', 'PAC_Values'});

        lme_spont_1 = fitlme(task_table_spont_1,'PAC_Values ~ Expected_Region + (1|Patient_ID)','FitMethod','ML');

    elseif j==4
        task_table_spont_2 = table(patient_ID_table(1:pac_patients*148), exp_region_table(1:pac_patients*148), ...
            PAC_values_table(1:pac_patients*148), ...
            'VariableNames', {'Patient_ID', 'Expected_Region', 'PAC_Values'});

        lme_spont_2 = fitlme(task_table_spont_2,'PAC_Values ~ Expected_Region + (1|Patient_ID)','FitMethod','ML');
    end
end




