import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
from answers.boxinfo import BoxInfo
# Define categories and mapping
CATEGORIES = ["spiking", "moving", "standing", "waiting", "blocking", "digging", "setting", "jumping", "falling"]
CATEGORY_TO_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

class VolleyballActionDataset(Dataset):
    def __init__(self, pickle_file, dataset_root, transform=None):
        """
        Custom Dataset for Volleyball Action Classification
        :param pickle_file: Path to the pickle annotation file.
        :param dataset_root: Directory containing original frames.
        :param transform: Torchvision transformations to apply.
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.data = []
        self.cache = {}

        # Load pickle file
        with open(pickle_file, "rb") as f:
            videos_annot = pickle.load(f)

        # Extract metadata from pickle annotations
        for videoId, video_data in videos_annot.items():
            for clipId, clip_data in video_data.items():
                for frameId, boxes in clip_data['frame_boxes_dct'].items():

                    frame_path = os.path.join(dataset_root, "videos", videoId, clipId, f"{frameId}.jpg")
                    if not os.path.exists(frame_path):
                        continue  # Skip missing frames

                    for box_info in boxes:
                        box = box_info.box
                        category = box_info.category

                        if category in CATEGORY_TO_INDEX:
                            self.data.append((frame_path, box, CATEGORY_TO_INDEX[category]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads a frame, crops the player, applies transformations, and returns (image, label).
        """
        frame_path, box, label = self.data[idx]

        # Check if the cropped image is already in the cache
        if idx in self.cache:
            cropped_img = self.cache[idx]
        else:
            # Load the full-frame image
            image = Image.open(frame_path).convert("RGB")

            # Extract bounding box
            x1, y1, x2, y2 = box
            cropped_img = image.crop((x1, y1, x2, y2))  # Crop the player

            # Store the cropped image in the cache
            self.cache[idx] = cropped_img

        # Apply transformations
        if self.transform:
            cropped_img = self.transform(cropped_img)

        return cropped_img, label


# 🔹 Default transformations (can be imported in another script)
default_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for ResNet50
    transforms.RandomHorizontalFlip(p=0.5),  # Augmentation: Random flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Augmentation: Color jitter
    transforms.RandomRotation(10),  # Augmentation: Slight rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
