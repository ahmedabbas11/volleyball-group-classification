import os

def running_on_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')

videos_folder = 'videos'
if running_on_kaggle():
    print("Running on Kaggle")
    dataset_root = '/kaggle/input/group-activity-recognition-volleyball'
    annotations_folder = '/volleyball_tracking_annotation'
    working_dir = '/kaggle/working'
else:
    print("Running locally")
    dataset_root = '/Users/ahmedabbas/Documents/deep-learning/vollyball_project/volleyball-dataset'
    annotations_folder = 'annotations'
    working_dir = '.'