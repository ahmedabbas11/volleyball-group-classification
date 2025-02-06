import pickle
import torch
from torch.utils.data import Dataset

class VolleyballDataset(Dataset):
    def __init__(self, pickle_file, transform=None, target_transform=None):
        # Load data from the pickle file
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)  # Assuming the pickle file contains a list or dictionary

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Access the data sample
        sample = self.data[idx]

        # Assuming the sample contains 'input' and 'label'
        input_data = sample['input']
        label = sample['label']

        # Apply transformations if provided
        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            label = self.target_transform(label)

        return input_data, label