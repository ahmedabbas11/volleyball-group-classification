
import torch
import torch.nn as nn
from torchvision import models
from answers.volleyball_annot_loader import working_dir
from b3.volleyball_action_dataloader import data_loader

# Hyperparameters
batch_size = 128
num_epochs = 5
learning_rate = 0.0001
snapshot_dir = f'{working_dir}/snapshots'
os.makedirs(snapshot_dir, exist_ok=True)

def train():
    # Load data
    trainloader, testloader, classes = data_loader(batch_size=batch_size)

    # Instantiate the model
    num_classes = len(classes)
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify output layer

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification problems
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_step = 0
    snapshot_path = 'snapshots/snapshot_latest.pt'

    if os.path.exists(snapshot_path):
        checkpoint = torch.load(snapshot_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step'] + 1
        print(f"Resumed from step {start_step}")
    else:
        print("Starting from scratch.")

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
            if i % 1000 == 0:
                snapshot_path = os.path.join(snapshot_dir, f'snapshot_step_{i}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': i,
                }, snapshot_path)
                print(f"Saved snapshot: {snapshot_path}")
                # Save snapshot again
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                }, 'snapshots/snapshot_latest.pt')

    print('Finished Training')

    # Save the trained model
    torch.save(model.state_dict(), 'b3_resnet50_stagA_player_pos.pth')
    print("Model saved as b3_resnet50_stagA_player_pos.pth")
