import os
import glob
import random
import shutil

import common

source_folder_path = './'
old_file_path_list = glob.glob(os.path.join(source_folder_path, '*.csv'))

random.seed(666)
random.shuffle(old_file_path_list)

for old_file_index, old_file_path in enumerate(old_file_path_list, start=1):
    new_file_name = 'Anonymous_' + str(old_file_index).zfill(2) + '.csv'
    new_file_path = os.path.join(common.SUBMISSIONS_FOLDER_PATH, new_file_name)
    shutil.copyfile(old_file_path, new_file_path)
