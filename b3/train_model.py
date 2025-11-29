import os
import torch
import time
import torch.nn as nn
from config.config import working_dir, output_dir
from b3.volleyball_player_dataloader import data_loader

# Hyperparameters
batch_size = 128
num_epochs = 5
learning_rate = 0.0001
snapshot_dir = f'{working_dir}/snapshots'
models_dir = f'{output_dir}/models'
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
latest_snapshot_path = os.path.join(output_dir, 'snapshots', 'snapshot_latest.pt')

def train():
    # Load data
    trainloader, valLoader, classes = data_loader(batch_size=batch_size)

    # Instantiate the model
    num_classes = len(classes)
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify output layer

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For classification problems
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_step = 0
    start_epoch = 0
    loss = 0.0

    if os.path.exists(latest_snapshot_path):
        checkpoint = torch.load(latest_snapshot_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_loss = checkpoint['loss']
        start_step = checkpoint['step'] + 1
        print(f"Resumed from step {start_step} of epoch {start_epoch} with loss {start_loss:.4f}")
    else:
        print("Starting from scratch.")

    # Training loop
    print('training started')
    full_start_train_time = time.time()
    for epoch in range(start_epoch, num_epochs):  # Loop over the dataset multiple times
        running_loss = 0.0
        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("traingLoader size: ", len(trainloader))
        epoch_start_train_time = time.time()
        for step in range(start_step, len(trainloader)):
            start_train_time = time.time()
            data = trainloader.__iter__().__next__()
            loading_time = time.time() - start_train_time
            print(f"get Item for step {step} took: {loading_time:.4f} seconds")
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
            if step % 20 == 19:  # Print every 200 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {step + 1}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0
            if step % 1000 == 0:
                save_checkpoint(model, optimizer, epoch, step, loss)
            train_time = time.time() - start_train_time
            print(f"Training step {step} time: {train_time:.4f} seconds")
        epoch_train_time = time.time() - epoch_start_train_time
        print(f"Epoch ended in {epoch_train_time:.4f} seconds")
    full_train_time = time.time() - full_start_train_time
    print(f"Finished Training in {full_train_time:.4f} seconds")

    # Save the trained model
    model_path = os.path.join(models_dir, 'b3_a_resnet50_player_pos.pth')
    torch.save(model.state_dict(), model_path)
    print("Model saved as b3_a_resnet50_player_pos.pth")

def save_checkpoint(model, optimizer, epoch, step, loss):
    snapshot_path = os.path.join(snapshot_dir, f'snapshot_epoch_{epoch}_step_{step}.pt')
    checkpoint ={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'step': step
                }
    torch.save(checkpoint, snapshot_path)
    print(f"Saved snapshot: {snapshot_path}")
    # Save latest snapshot
    torch.save(checkpoint, latest_snapshot_path)
