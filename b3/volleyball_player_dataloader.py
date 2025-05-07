from pandas.core.dtypes.common import classes
from torch.utils.data import DataLoader, random_split

from config.config import dataset_root, videos_folder, annotations_folder, working_dir
from b3.volleyball_player_dataset import VolleyballPlayerDataset, default_transform, CATEGORIES
from config.config import video_splits


def data_loader(batch_size=32, num_workers=4):
    # 🔹 Paths
    PICKLE_FILE = f'{working_dir}/annot_all.pkl'

    # 🔹 Load train dataset
    trainDataset = VolleyballPlayerDataset(
        pickle_file=PICKLE_FILE,
        dataset_root=dataset_root,
        videos_folder = videos_folder,
        splits=video_splits['train'],
        transform=default_transform)

    # 🔹 Print dataset info
    print(f"Training Dataset Size: {len(trainDataset)} samples")

    # 🔹 Load train dataset
    validationDataset = VolleyballPlayerDataset(
        pickle_file=PICKLE_FILE,
        dataset_root=dataset_root,
        videos_folder = videos_folder,
        splits=video_splits['validation'],
        transform=default_transform)

    # 🔹 Print dataset info
    print(f"Validation Dataset Size: {len(trainDataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validationDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = CATEGORIES
    print(f'classes calculated {classes}')
    return train_loader, val_loader, classes