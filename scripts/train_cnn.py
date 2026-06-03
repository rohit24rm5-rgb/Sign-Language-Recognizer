import sys
import os
# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import os

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Define transforms with Data Augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation uses standard transforms without augmentation
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} directory not found.")
        return

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    # Create two datasets to apply different transforms
    full_dataset_train = datasets.ImageFolder(data_dir, transform=train_transforms)
    full_dataset_val = datasets.ImageFolder(data_dir, transform=val_transforms)
    
    class_names = full_dataset_train.classes
    print(f"Classes found ({len(class_names)}): {class_names}")

    # Split dataset
    train_size = int(0.8 * len(full_dataset_train))
    val_size = len(full_dataset_train) - train_size
    
    # We use torch.manual_seed to ensure same split for train and val datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_dataset_train, [train_size, val_size], generator=generator)
    
    generator = torch.Generator().manual_seed(42)
    _, val_dataset = random_split(full_dataset_val, [train_size, val_size], generator=generator)
    
    print(f"Training set: {train_size} images")
    print(f"Validation set: {val_size} images")

    # Dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize ResNet18 model with Transfer Learning (pre-trained weights)
    print("Initializing ResNet18 with IMAGENET1K_V1 pre-trained weights...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    epochs = 5
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Step the scheduler
        exp_lr_scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"==> Epoch [{epoch+1}/{epochs}] Validation Accuracy: {val_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Train Loss: {running_loss/len(train_loader):.4f}")

    # Save trained weights
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "cnn_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Fine-tuning complete! Optimized model weights saved to {save_path}")

if __name__ == '__main__':
    main()
