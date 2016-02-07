from joblib import Parallel, delayed
from sklearn.cross_validation import KFold
import numpy as np
import prepare_data
import solution_basic

# Get image paths in the training and testing datasets
image_paths_in_training_dataset, training_image_index_list = prepare_data.get_image_paths_in_training_dataset()

def inspect_final_data_set_without_labels(image_index_list, seed):
    np.random.seed(seed)
    image_index_array = np.array(image_index_list)

    # Cross Validation
    fold_num = 5
    label_kfold = KFold(image_index_array.size, n_folds=fold_num, shuffle=True)

    true_records_num_list = []
    false_records_num_list = []

    for _, fold_item in enumerate(label_kfold):
        # Generate final data set
        selected_index_array = image_index_array[fold_item[0]]
        _, Y_train = solution_basic.get_record_map(selected_index_array, None)

        true_records = Y_train == 1
        true_records_num = np.sum(true_records)
        false_records_num = Y_train.size - true_records_num

        true_records_num_list.append(true_records_num)
        false_records_num_list.append(false_records_num)

    return (true_records_num_list, false_records_num_list)

def inspect_final_data_set_with_labels(image_index_list, seed):
    np.random.seed(seed)

    # Cross Validation
    fold_num = 5
    unique_label_values = np.unique(image_index_list)
    selected_label_values = np.random.choice(unique_label_values, size=np.ceil(unique_label_values.size * (fold_num - 1) / fold_num), replace=False)

    selected_index_list = []
    for single_image_index in image_index_list:
        if single_image_index in selected_label_values:
            selected_index_list.append(single_image_index)
    selected_index_array = np.array(selected_index_list)

    _, Y_train = solution_basic.get_record_map(selected_index_array, None)

    true_records = Y_train == 1
    true_records_num = np.sum(true_records)
    false_records_num = Y_train.size - true_records_num

    return ([true_records_num], [false_records_num])

repeated_num = 20
seed_array = np.random.choice(range(repeated_num), size=repeated_num, replace=False)
# records_list = (Parallel(n_jobs=-1)(delayed(inspect_final_data_set_without_labels)(training_image_index_list, seed) for seed in seed_array))
records_list = (Parallel(n_jobs=-1)(delayed(inspect_final_data_set_with_labels)(training_image_index_list, seed) for seed in seed_array))

true_records_num_list = []
false_records_num_list = []

for single_true_records_num_list, single_false_records_num_list in records_list:
    for value in single_true_records_num_list:
        true_records_num_list.append(value)

    for value in single_false_records_num_list:
        false_records_num_list.append(value)

for single_list in [true_records_num_list, false_records_num_list]:
    print("The min is {:d}. The max is {:d}. The mean is {:.4f}".format(np.min(single_list), np.max(single_list), np.mean(single_list)))