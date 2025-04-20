from pandas.core.dtypes.common import classes
from torch.utils.data import DataLoader, random_split

from answers.volleyball_annot_loader import working_dir, dataset_root, videos_folder
from b3.volleyball_action_dataset import VolleyballActionDataset, default_transform, CATEGORIES


def data_loader(batch_size=32, num_workers=4):
    # 🔹 Paths
    PICKLE_FILE = f'{working_dir}/annot_all.pkl'

    # 🔹 Load dataset
    dataset = VolleyballActionDataset(pickle_file=PICKLE_FILE, dataset_root=dataset_root, videos_folder = videos_folder, transform=default_transform)

    # 🔹 Print dataset info
    print(f"Dataset Size: {len(dataset)} samples")

    # Define split ratios
    train_ratio = 0.8  # 80% training, 20% testing
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Number of unique videos (or another label strategy)
    print(f'dataset Loaded {len(dataset)}')
    classes = CATEGORIES
    print(f'classes calculated {classes}')
    return train_loader, test_loader, classes