import os
import bson

# Dataset
PROJECT_NAME = "Cdiscount Image Classification"
PROJECT_FOLDER_PATH = os.path.join(os.path.expanduser("~"), "Documents/Dataset", PROJECT_NAME)
VANILLA_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "vanilla")
TRAIN_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "train.bson")
TEST_FILE_PATH = os.path.join(VANILLA_DATASET_FOLDER_PATH, "test.bson")
EXTRACTED_DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "extracted")
TRAIN_FOLDER_PATH = os.path.join(EXTRACTED_DATASET_FOLDER_PATH, "train")
TEST_FOLDER_PATH = os.path.join(EXTRACTED_DATASET_FOLDER_PATH, "test")

def run():
    for dataset_file_path, dataset_folder_path in zip((TRAIN_FILE_PATH, TEST_FILE_PATH), (TRAIN_FOLDER_PATH, TEST_FOLDER_PATH)):
        print("Processing {} ...".format(dataset_file_path))

        with open(dataset_file_path, "rb") as dataset_file_object:
            data_generator = bson.decode_file_iter(dataset_file_object)

            for data in data_generator:
                category_id = data.get("category_id", "dummy")
                category_folder_path = os.path.join(dataset_folder_path, str(category_id))
                os.makedirs(category_folder_path, exist_ok=True)

                product_id = data["_id"]
                for picture_id, picture_dict in enumerate(data["imgs"]):
                    picture_content = picture_dict["picture"]
                    picture_file_path = os.path.join(category_folder_path, "{}_{}.jpg".format(product_id, picture_id))
                    with open(picture_file_path, "wb") as picture_file_object:
                        picture_file_object.write(picture_content)

    print("All done!")

if __name__ == "__main__":
    run()
