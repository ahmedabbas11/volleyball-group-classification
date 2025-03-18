import numpy as np
import torch
from torch import nn
from torchvision import models

from b1.data_loader import get_dataset


def test_model(model_path, batch_size=32):
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load test data
    _, testloader, classes = get_dataset(batch_size=batch_size)

    # Load the trained model
    print('loading model')
    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, len(classes))  # Modify output layer
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    net.eval()  # Set the model to evaluation mode
    print('test started')
    # Testing loop
    correct = 0
    total = 0
    misclassified = []
    counter = 1
    with torch.no_grad():
        for data in testloader:
            print(f'Testing on batch {counter}')
            counter+=1
            images, labels, paths = data[0].to(device), data[1].to(device), data[2]
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    print(f'Misclassified image {paths[i]} and label {labels[i]} and predicted {predicted[i]}')
                    misclassified.append((images[i], labels[i], predicted[i]))

    # Print accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

    # Visualize some misclassified images
    for img, true_label, pred_label in misclassified[:5]:
        plt.imshow(img.cpu().permute(1, 2, 0))
        plt.title(f"True: {classes[true_label]} | Pred: {classes[pred_label]}")
        plt.show()


import matplotlib.pyplot as plt
import torchvision

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualize_predictions(net, testloader, classes, device):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Move data to device
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Display images with predictions
    imshow(torchvision.utils.make_grid(images.cpu()))
    print('GroundTruth:', ' '.join(f'{classes[labels[j]]}' for j in range(4)))
    print('Predicted:  ', ' '.join(f'{classes[predicted[j]]}' for j in range(4)))

# Add visualization to the main function
# if __name__ == "__main__":
#     model_path = 'cnn_cifar10.pth'
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     net = SimpleCNN()
#     net.load_state_dict(torch.load(model_path))
#     net.to(device)
#     _, testloader, classes = get_cifar10_loaders(batch_size=32)
#     visualize_predictions(net, testloader, classes, device)

if __name__ == "__main__":
    # Path to the trained model
    model_path = '../trial1_resnet50_middle_frame.pth'  # Update with your actual model path
    # with open(model_path, 'rb') as file:
    #     print(f'file {file.name}')
    test_model(model_path)
