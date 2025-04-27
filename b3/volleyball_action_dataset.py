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
    def __init__(self, pickle_file, dataset_root, videos_folder, transform=None):
        """
        Custom Dataset for Volleyball Action Classification
        :param pickle_file: Path to the pickle annotation file.
        :param dataset_root: Directory containing original frames.
        :param transform: Torchvision transformations to apply.
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.data = []
        self.last_frame_id = None
        self.last_frame_image = None

        # Load pickle file
        with open(pickle_file, "rb") as f:
            print(f"Loading pickle file: {pickle_file}")
            videos_annot = pickle.load(f)

        print(f"videos_annot size: {len(videos_annot)}")
        print(f"dataset_root: {dataset_root}")
        print(f"videos folder: {os.path.join(dataset_root, videos_folder)}")
        # Extract metadata from pickle annotations
        for videoId, video_data in videos_annot.items():
            print(f"video data size: {len(video_data)}")
            for clipId, clip_data in video_data.items():
                for frameId, boxes in clip_data['frame_boxes_dct'].items():

                    frame_path = os.path.join(dataset_root, videos_folder, videoId, clipId, f"{frameId}.jpg")
                    # Check if the frame exists
                    if not os.path.exists(frame_path):
                        print(f"Frame not found: {frame_path}")
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
        # Only load frame if different from last one
        if self.last_frame_id != frame_path:
            # Load the full-frame image
            # print(f"Loading frame: {frame_path}")
            self.last_frame_image = Image.open(frame_path).convert("RGB")
            self.last_frame_id = frame_path

        # Extract bounding box
        x1, y1, x2, y2 = box
        cropped_img = self.last_frame_image.crop((x1, y1, x2, y2))  # Crop the player

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
