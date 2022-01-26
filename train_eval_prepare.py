from my_utils import split_data

if __name__ == "__main__":

    path_to_data = "E:\\Projects\\Python\\InsectsRecognition\\data\\Train" #path to Main TRAIN Folder
    path_to_save_train = "E:\\Projects\\Python\\InsectsRecognition\\data\\training_data\\train" # Path to save the train set
    path_to_save_val = "E:\\Projects\\Python\\InsectsRecognition\data\\training_data\\val" # Path to save the val set

    split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)