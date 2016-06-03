close all;
clear;
clc;

%% Load data set
disp('Loading data set...');

% Define directories
[current_path, ~, ~] = fileparts(mfilename('fullpath'));
data_path = fullfile(current_path, 'input');
libsvm_path = fullfile(current_path, 'libsvm\windows');

% Add libsvm library
addpath(libsvm_path);

% Load training data set
training_features = csvread(fullfile(data_path, 'X_train.csv'), 1, 1);
training_labels = csvread(fullfile(data_path, 'y_train.csv'), 1, 1);

% Load testing data set
testing_features = csvread(fullfile(data_path, 'X_test.csv'), 1, 1);

disp('Loading data set successfully.');

%% Training/Testing phase
disp('Training/Testing model...');

CV_num = 11;
fold_num = 5;
testing_labels_pack = cell(CV_num, fold_num);
testing_labels_base = zeros(size(testing_features, 1), size(training_labels, 2));

tic;
parpool(2);
parfor CV_index = 1:CV_num
    fprintf('Training phase %d/%d...\n', CV_index, CV_num);
    
    cvInd = crossvalind('Kfold', size(training_features, 1), fold_num);
    for fold_index = 1:fold_num
        selected_record_flag = (cvInd ~= fold_index);
        testing_labels_pack{CV_index, fold_index} = testing_labels_base;
        
        % Estimate the angles separately
        for angle_index = 1:size(training_labels, 2)
            model = svmtrain(training_labels(selected_record_flag, angle_index), ...
                training_features(selected_record_flag, :), '-t 0 -q');
            testing_label_vector = zeros(size(testing_features, 1), 1);
            testing_labels_pack{CV_index, fold_index}(:, angle_index) = ...
                svmpredict(testing_label_vector, testing_features, model, '-q');
        end
    end
end
delete(gcp);
elapsed_time = toc;
fprintf('Spend %.2f seconds on Training/Testing.\n', elapsed_time);

%% Generate final prediction
testing_labels_store = cell(size(testing_features, 1), size(training_labels, 2));
for CV_index = 1:CV_num
    for fold_index = 1:fold_num
        for testing_record_index = 1:size(testing_features, 1)
            for angle_index = 1:size(training_labels, 2)
                testing_labels_pack_temp = testing_labels_pack{CV_index, fold_index};
                testing_labels_store{testing_record_index, angle_index} = ...
                    [testing_labels_store{testing_record_index, angle_index} ...
                    testing_labels_pack_temp(testing_record_index, angle_index)];
            end
        end
    end
end

testing_labels_mean = testing_labels_base;
testing_labels_median = testing_labels_base;
testing_labels_mode = testing_labels_base;
for testing_record_index = 1:size(testing_features, 1)
    for angle_index = 1:size(training_labels, 2)
        testing_labels_store_temp = testing_labels_store{testing_record_index, angle_index};
        testing_labels_mean(testing_record_index, angle_index) = mean(testing_labels_store_temp);
        testing_labels_median(testing_record_index, angle_index) = median(testing_labels_store_temp);
        testing_labels_mode(testing_record_index, angle_index) = mode(testing_labels_store_temp);
    end
end

%% Create submission files
disp('Creating submission files...');
create_submission_file(testing_labels_mean, 'submission_mean.csv');
create_submission_file(testing_labels_median, 'submission_median.csv');
create_submission_file(testing_labels_mode, 'submission_mode.csv');
disp('Submission files created.');
