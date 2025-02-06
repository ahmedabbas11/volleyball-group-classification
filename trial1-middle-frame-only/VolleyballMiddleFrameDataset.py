import pickle

from PIL import Image
from torch.utils.data import Dataset


class VolleyballMiddleFrameDataset(Dataset):
    def __init__(self, pickle_file, transform=None, target_transform=None):
        # Load data from the pickle file
        with open(pickle_file, 'rb') as file:
            self.videos_annot = pickle.load(file)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.videos_annot)

    def __getitem__(self, idx):
        img_path, label = self.videos_annot[idx]
        print(f'img_path -- {img_path} -- label {label}')
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label