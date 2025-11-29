import os
from config.config import dataset_root, videos_folder, annotations_folder, working_dir
from b3.volleyball_player_dataset import VolleyballPlayerDataset, default_transform, CATEGORIES
from config.config import video_splits
from albumentations.pytorch import ToTensorV2
import albumentations as A


# Hyperparameters
batch_size = 128
# num_epochs = 5
# learning_rate = 0.0001
# snapshot_dir = f'{working_dir}/snapshots/b3_b'
# models_dir = f'{output_dir}/models/b3_b'
# os.makedirs(snapshot_dir, exist_ok=True)
# os.makedirs(models_dir, exist_ok=True)
# latest_snapshot_path = os.path.join(output_dir, 'snapshots', 'b3_b', 'snapshot_b3_b_latest.pt')

def extract_features():
    # Load data
    trainloader = data_loader(batch_size=batch_size)
    # todo we need in the datasouorce the frame labels

    # load model from stage a
    # for each frame in the trainloader
    # extract features for all the cropped images 
    # then max pool the players
    
    # stage C
    # build simple classifier
    # classify over group activity(8 classes) and the input will be the feature you extracted and pooled from stage B.




def data_loader(batch_size=32, num_workers=4):
    # 🔹 Paths
    PICKLE_FILE = f'{working_dir}/annot_all.pkl'

    # 🔹 Load train dataset
    trainDataset = VolleyballPlayerDataset(
        pickle_file=PICKLE_FILE,
        dataset_root=dataset_root,
        videos_folder = videos_folder,
        splits=video_splits['train'],
        transform=train_transforms)
    
    # 🔹 Print dataset info
    print(f"Training Dataset Size: {len(trainDataset)} samples")

    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader


