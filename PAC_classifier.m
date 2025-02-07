% -----------------------------------------------------------------------------------------------------------
% Patient-level PAC analysis using Classification
% Written by Srijita Das on June 2024
% Modified on August 17, 2024

%%
clear

%% Step 1: Define input data
% Addpath
addpath("U:\shared\users\sdas\brainstorm_db\PAC_current\tensor_toolbox-v3.6");

% Load file containing scout names
load("Destrieux_row_names.mat");

% Define the base directory where your data is stored
base_dir = 'U:\shared\users\sdas\meg-UNMC_results\';

% Define tasks and sides
tasks = {'SEF','SEF'};                   %only SEF will be used for classifier
datafile_name = {'SEFul','SEFur'};

% Get the list of patients in base directory
patients_all = dir(base_dir);
patient_names = {patients_all([patients_all.isdir]).name};
patient_names = patient_names(~ismember(patient_names, {'.','..'}));

%% Step 2: Initialize variables
% 1st 4400 values ul, next 4400 values ur
pt = 30; % number of patients
num_scouts = 148;
num_tasks = length(tasks);
num_modes = 2; % number of modes after tensor decomposition
total_samples = num_scouts * pt * num_tasks;
PAC_values_matrix = zeros(total_samples, num_modes);      %for all scouts and tasks
expected_regions_vector = false(total_samples, 1);
%patient_ID_table = repelem((1:pt)', 148, 1);
%patient_ID_vector = cat(1, patient_ID_table, patient_ID_table);

% Initialize list to store patient names with PAC data (preallocate size)
patients_with_pac = cell(pt, 1);
pac_patient_count = 0;
patient_idx_map = containers.Map;

%% Step 3: Tensor decomposition and PAC

% Loop through each task
for j = 1:length(tasks)
    row_idx = 0;
    % Loop through each patient with PAC data
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
                row_idx = row_idx + 1;
            end
        end

        if has_pac && row_idx <= pt
            % Add patient to the list if they have PAC data
            if ~isKey(patient_idx_map, subjectID)
                pac_patient_count = pac_patient_count + 1;
                patients_with_pac{pac_patient_count} = subjectID;
                patient_idx_map(subjectID) = pac_patient_count;
            end

            % Load Direct PAC data
            sPAC = m.sPAC;
            direct_PAC = sPAC.DirectPAC;

            direct_PAC_reshaped = reshape(direct_PAC, [], 8, 81); % converting into 3D matrix to remove time component

            % Step 3: Decompose tensor
            direct_PAC_3D_tensor = tensor(direct_PAC_reshaped);

            % Decomposition on 3-D tensor for each rank
            rank = 2;
            PAC = cp_als(direct_PAC_3D_tensor, rank,'tol',1e-6,'maxiters',1000);

            start_idx = ((j - 1) * pt * num_scouts) + ((row_idx - 1) * num_scouts) + 1; 
            end_idx = start_idx + num_scouts - 1;  % 1st 148 rows 1st patient, from 149 2nd patient...

            PAC_values_matrix(start_idx:end_idx, :) = PAC.U{1}; % Store all PAC values
            %PAC_values_matrix = zscore(PAC_values_matrix);

            if j == 1
                exp_indices = [52, 56, 58, 90, 134, 138];
            elseif j == 2
                exp_indices = [51, 55, 57, 89, 133, 137];
            end

            expected_regions_vector(start_idx:end_idx) = ismember(1:num_scouts, exp_indices);
        end
    end
end

% Trim the preallocated cell array to the actual number of patients with PAC data
patients_with_pac = patients_with_pac(1:pac_patient_count);

%% Step 4: Generate cross-validation indices

% Generate 10-fold cross-validation indices for patients
patient_fold_indices = crossvalind('Kfold', pt, 10); % 10-fold cross-validation indices for patients

% Repeat the indices for each scout and task
group_ID_vector = repelem(patient_fold_indices, num_scouts);
group_ID_vector = repmat(group_ID_vector, num_tasks, 1);

%% Step 5: Train and test the SVM classifier

% Define the number of folds
num_folds = 10;                             %10-fold cross-validation
CV_results = false(total_samples, 1);
CV_scores = zeros(total_samples, 1);


% Loop through each fold
for fold = 1:num_folds                        
    % Create training and validation indices
    train_idx = group_ID_vector ~= fold;
    test_idx = group_ID_vector == fold;

    % Define training and testing data
    X_train = PAC_values_matrix(train_idx, :);
    y_train = expected_regions_vector(train_idx);
    X_test = PAC_values_matrix(test_idx, :);
    y_test = expected_regions_vector(test_idx);

    % figure;
    % gscatter(X_train(:,1),X_train(:,2),y_train)

    rng default
    % Train the SVM model
    SVMModel = fitcsvm(X_train, y_train,"KernelFunction","rbf",'Standardize',true);         
    %score_SVM = fitPosterior(SVMModel);

    % Predict on the test set
    [predictions, score] = predict(SVMModel, X_test);

    % Store the results of scores
    CV_results(test_idx) = predictions;
    CV_scores(test_idx) = score(:, 2);
end

% Create confusion matrix
TP = sum(CV_results == 1 & expected_regions_vector == 1); % True Positives
TN = sum(CV_results == 0 & expected_regions_vector == 0); % True Negatives
FP = sum(CV_results == 1 & expected_regions_vector == 0); % False Positives
FN = sum(CV_results == 0 & expected_regions_vector == 1); % False Negatives

%C = confusionmat(expected_regions_vector,CV_results);
C = [TP, FN; FP, TN];
Cm = confusionchart(C);

% Calculate evaluation metrics
specificity = TN / (FP + TN);
fpr = FP / (FP + TN);
ppv = TP/ (TP + FP);

% Calculate AUC-ROC
[X, Y, T, AUC] = perfcurve(expected_regions_vector, CV_scores, 1);    % true class labels compared to classifier results
figure;
plot(X, Y);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve');
fprintf('AUC: %.2f\n', AUC);

%% Cross-check results
% Step 1: Identify true positive & False Positive Indices
tp_id = find(CV_results == 1 & expected_regions_vector == 1);
fp_id = find(CV_results == 1 & expected_regions_vector == 0);  % indices which were predicted but were not expected

% Step 2: Initialize variables to store true & false positive details
tp_patients = zeros(length(tp_id), 1);
tp_tasks = strings(length(tp_id), 1);
tp_brain_regions = zeros(length(tp_id), 1);

fp_patients = zeros(length(fp_id), 1);
fp_tasks = strings(length(fp_id), 1);
fp_brain_regions = zeros(length(fp_id), 1);

% Step 3: Map true and False Positive Indices to Patient, Task, and Brain Region

% True Positives
for i = 1:length(tp_id)
    t_idx = tp_id(i);
    tp_task_idx = ceil(t_idx / (pt * num_scouts));  % Task index
    tp_tasks(i) = datafile_name{tp_task_idx};  % Task name
    
    % Determine the patient index within the task
    patient_within_task_idx = ceil((t_idx - (tp_task_idx - 1) * pt * num_scouts) / num_scouts);
    tp_patients(i) = patient_within_task_idx;
    
    % Determine the brain region index within the patient
    tp_brain_region_idx = mod(t_idx - 1, num_scouts) + 1;
    tp_brain_regions(i) = tp_brain_region_idx;
end

% False Positives
for k = 1:length(fp_id)
    f_idx = fp_id(k);
    fp_task_idx = ceil(f_idx / (pt * num_scouts));  % Task index
    fp_tasks(k) = datafile_name{fp_task_idx};  % Task name
    
    % Determine the patient index within the task
    patient_within_task_idx = ceil((f_idx - (fp_task_idx - 1) * pt * num_scouts) / num_scouts);
    fp_patients(k) = patient_within_task_idx;
    
    % Determine the brain region index within the patient
    brain_region_idx = mod(f_idx - 1, num_scouts) + 1;
    fp_brain_regions(k) = brain_region_idx;
end

% Step 4: Plot the results

% Plot 1: False Positives per Patient
figure;
histogram(fp_patients, 1:length(patients_with_pac)+1,'FaceColor','r');
xlabel('Patient ID');
ylabel('Number of False Positives');
title('False Positives per Patient');
xticks(1:length(patients_with_pac));
xticklabels(patients_with_pac);
xtickangle(45);
grid on;

% Plot 2: False Positives per Task
figure;
histogram(categorical(fp_tasks), 'Categories', datafile_name,'FaceColor','r');
xlabel('Task');
ylabel('Number of False Positives');
title('False Positives per Task');
grid on;

% Plot 3: False Positives per Brain Region
figure;
histogram(fp_brain_regions, 1:num_scouts+1);
xlabel('Brain Region');
ylabel('Number of False Positives');
title('False Positives per Brain Region');
xticks(1:num_scouts);
grid on;

% Plot 4: True Positives per Patient
figure;
histogram(tp_patients, 1:length(patients_with_pac)+1);
xlabel('Patient ID');
ylabel('Number of True Positives');
title('True Positives per Patient');
xticks(1:length(patients_with_pac));
xticklabels(patients_with_pac);
xtickangle(45);
grid on;

% Plot 5: True Positives per Task
figure;
histogram(categorical(tp_tasks), 'Categories', datafile_name);
xlabel('Task');
ylabel('Number of True Positives');
title('True Positives per Task');
grid on;

% Plot 6: True Positives per Brain Region
figure;
histogram(tp_brain_regions, 1:num_scouts+1);
xlabel('Brain Region');
ylabel('Number of True Positives');
title('True Positives per Brain Region');
xticks(1:num_scouts);
grid on;

