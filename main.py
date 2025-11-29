from b3.test_pickle_file import test_pickle
from b3.train_model import train
from answers.boxinfo import BoxInfo
from config.config import dataset_root, videos_folder, annotations_folder, working_dir

if __name__ == "__main__":
    test_pickle(dataset_root)