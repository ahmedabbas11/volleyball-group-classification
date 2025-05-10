from pandas.core.dtypes.common import classes
from torch.utils.data import DataLoader, random_split

from config.config import dataset_root, videos_folder, annotations_folder, working_dir
from b3.volleyball_player_dataset import VolleyballPlayerDataset, default_transform, CATEGORIES
from config.config import video_splits
from albumentations.pytorch import ToTensorV2
import albumentations as A


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

    # 🔹 Load train dataset
    validationDataset = VolleyballPlayerDataset(
        pickle_file=PICKLE_FILE,
        dataset_root=dataset_root,
        videos_folder = videos_folder,
        splits=video_splits['validation'],
        transform=val_transforms)

    # 🔹 Print dataset info
    print(f"Validation Dataset Size: {len(validationDataset)} samples")

    # Create DataLoaders
    train_loader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validationDataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    classes = CATEGORIES
    print(f'classes calculated {classes}')
    return train_loader, val_loader, classes


train_transforms = A.Compose([
        A.Resize(224, 224),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.ColorJitter(brightness=0.2),
            A.RandomBrightnessContrast(),
            A.GaussNoise()
        ], p=0.5),
        A.OneOf([
            A.HorizontalFlip(),
            A.VerticalFlip(),
        ], p=0.05),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

val_transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])