function [] = create_submission_file(testing_labels, file_name)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
narginchk(2, 2);

file_ID = fopen(file_name, 'w');
fprintf(file_ID, 'Id,Angle1,Angle2\n');
for testing_record_index = 1:size(testing_labels, 1)
    fprintf(file_ID, '%d,%.4f,%.4f\n', testing_record_index + 1953, testing_labels(testing_record_index, 1), testing_labels(testing_record_index, 2));
end
fclose(file_ID);
end
