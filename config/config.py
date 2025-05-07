import os

def running_on_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')

videos_folder = 'videos'
if running_on_kaggle():
    print("Running on Kaggle")
    dataset_root = '/kaggle/input/group-activity-recognition-volleyball'
    annotations_folder = '/volleyball_tracking_annotation'
    working_dir = '/kaggle/working'
    output_dir = '/kaggle/outputs/'
else:
    dataset_root = '/Users/ahmedabbas/Documents/deep-learning/vollyball_project/volleyball-dataset'
    annotations_folder = 'annotations'
    working_dir = '.'
    output_dir = '.'



video_splits = {
    'train': [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    'validation': [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    'test': [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
}