import os
import shutil
import numpy as np


def split_dataset_into_test_and_train_sets(all_data_dir, testing_data_dir, testing_data_pct):
    """
    Recreate testing and training directories
    Ref:https://github.com/keras-team/keras/issues/5862
    """
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    num_testing_files = 0

    files_count = {}

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        testing_data_category_dir = testing_data_dir + '/' + category_name


        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct and files_count.get(category_name, 0) < 20:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
                files_count[category_name] = files_count.get(category_name, 0) + 1


if __name__ == "__main__":
    all_images = "data"
    test = "validate"

    split_dataset_into_test_and_train_sets(all_images,test,0.2)
