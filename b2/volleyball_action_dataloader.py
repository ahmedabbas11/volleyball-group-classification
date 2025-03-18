import torch
from torch.utils.data import DataLoader

from answers.volleyball_annot_loader import dataset_root
from b2.volleyball_action_dataset import VolleyballActionDataset, default_transform

def data_loader():
    # 🔹 Paths
    PICKLE_FILE = f'{dataset_root}/annot_all.pkl'

    # 🔹 Load dataset
    dataset = VolleyballActionDataset(pickle_file=PICKLE_FILE, dataset_root=dataset_root, transform=default_transform)

    # 🔹 Create DataLoader
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

    # 🔹 Print dataset info
    print(f"Dataset Size: {len(dataset)} samples")

    # 🔹 Check a batch
    sample_batch = next(iter(train_loader))
    print(f"Batch Shape: {sample_batch[0].shape}, Labels: {sample_batch[1]}")