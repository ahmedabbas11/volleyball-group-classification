
import torch
import torch.nn as nn
from torchvision import models

from b1.data_loader import get_dataset

# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001

# Load data
trainloader, testloader, classes = get_dataset(batch_size=batch_size)

# Print class names
print("Classes:", classes)
# Instantiate the model
num_classes = len(classes)
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify output layer

# Move model to GPU if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For classification problems
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
print('training started')
for epoch in range(num_epochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # Get inputs and labels
        image, label = data
        images, labels = image.to(device), label.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 20 == 19:  # Print every 200 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 20:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'trial1_resnet50_middle_frame.pth')
print("Model saved as trial1_resnet50_middle_frame.pth")
