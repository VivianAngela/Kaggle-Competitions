close all;
clear;
clc;

%% Define paths

% Remember to edit the following variable.
% Now we assume that current folder has subfolders "train" and "test".
data_path = '.';

training_folder_name = 'train';
testing_folder_name = 'test';
pairs_file_name = 'pairs.csv';

training_folder_path = fullfile(data_path, training_folder_name);
testing_folder_path = fullfile(data_path, testing_folder_name);
pairs_file_path = fullfile(data_path, pairs_file_name);

%% Load all deep features for all test files into memory
% The data is stored into a "map" structure, which can be accessed
% via the filename: features = testing_deepfeatures_map(filename);

testing_deepfeatures_file_name_list = dir(fullfile(testing_folder_path, '*_deepfeatures.csv'));
testing_deepfeatures_file_name_list = {testing_deepfeatures_file_name_list.name}';
testing_deepfeatures_map = containers.Map;

for testing_deepfeatures_index = 1:length(testing_deepfeatures_file_name_list)
    
    testing_deepfeatures_file_name = testing_deepfeatures_file_name_list{testing_deepfeatures_index};
    testing_deepfeatures_file_path = fullfile(testing_folder_path, testing_deepfeatures_file_name);
    
    deepfeatures = csvread(testing_deepfeatures_file_path);
    deepfeatures = deepfeatures';
    
    splitted_file_name = strsplit(testing_deepfeatures_file_name, '_');
    testing_image_file_name = splitted_file_name{1};
    
    testing_deepfeatures_map(testing_image_file_name) = deepfeatures;
    
    if rem(testing_deepfeatures_index, 100) == 0
        fprintf('%d/%d files loaded...\n', ...
                testing_deepfeatures_index, ...
                length(testing_deepfeatures_file_name_list));
    end
end

fprintf('All files loaded. Computing distances for pairs.\n');

%% Compute Euclidean distances for all requested image pairs
% Read the "pairs.csv" file

pairs_file_ID = fopen(pairs_file_path);
pairs_file_content = textscan(pairs_file_ID, '%d %s %s', 'HeaderLines', 1, 'Delimiter', ',');
fclose(pairs_file_ID);

Id_list = pairs_file_content{1};
file_1_list = pairs_file_content{2};
file_2_list = pairs_file_content{3};

% Store all distances here
distances = zeros(size(Id_list));

for distance_index = 1:length(distances)
    % Find the deep features related to the two files
    deepfeatures_1 = testing_deepfeatures_map(file_1_list{distance_index});
    deepfeatures_2 = testing_deepfeatures_map(file_2_list{distance_index});
    
    % Compute the distance
    deepfeatures_difference = deepfeatures_1 - deepfeatures_2;
    deepfeatures_distance = sqrt(sum(deepfeatures_difference .^ 2));
    distances(distance_index) = deepfeatures_distance;
    
    if rem(distance_index, 1000) == 0
        fprintf('%d/%d pairs done...\n', distance_index, length(distances));
    end
end

%% Create submission file
% Find largest distance
max_distance = max(distances);

% Compute similarity scores
similarity_scores = (max_distance - distances) / max_distance;

% Write similarity scores to file
submission_file_ID = fopen('submission.csv', 'w');
fprintf(submission_file_ID, 'Id,Prediction\n');

for similarity_scores_index = 1:length(similarity_scores)
    fprintf(submission_file_ID, '%d,%.6f\n', Id_list(similarity_scores_index), similarity_scores(similarity_scores_index));
end

fclose(submission_file_ID);

fprintf('Prediction finished.\n');
