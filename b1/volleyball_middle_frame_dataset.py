import pickle

from PIL import Image
from torch.utils.data import Dataset

class_labels = {
    'l_set': 0, 'l_spike': 1, 'l-spike': 1, 'r_set': 2, 'r_winpoint': 3,
    'r_spike': 4, 'l_winpoint': 5, 'l-pass': 6, 'r-pass': 7
}

class VolleyballMiddleFrameDataset(Dataset):
    def __init__(self, pickle_file, transform=None, target_transform=None):
        # Load data from the pickle file
        with open(pickle_file, 'rb') as file:
            self.videos_annot = pickle.load(file)
        print(f'Loaded {pickle_file}')
        print(f'pickle file {len(self.videos_annot)}')
        classes = set(label['label'] for label in self.videos_annot.values())
        print(f'classes {classes}')
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        for  _, entry in self.videos_annot.items():
            img_path = entry['img_path']
            lbl_str = entry['label']
            self.data.append((img_path, class_labels[lbl_str]))
        print('data loaded')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # Apply transformations if provided
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label, img_path
