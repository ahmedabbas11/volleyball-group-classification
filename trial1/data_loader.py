from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms

from trial1.volleyball_middle_frame_dataset import VolleyballMiddleFrameDataset
from trial1.volleyball_middle_frame_image_loader import dataset_root


def get_dataset(batch_size=32):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    picle_file = f'{dataset_root}/annot_middle_frame.pkl'
    dataset = VolleyballMiddleFrameDataset(picle_file, transform=transform)
    # Define split ratios
    train_ratio = 0.8  # 80% training, 20% testing
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    # Split dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # Number of unique videos (or another label strategy)
    classes = set(label for _, label in train_set.dataset)

    return train_loader, test_loader, classes