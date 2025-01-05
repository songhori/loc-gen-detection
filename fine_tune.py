import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#######################################################

# Load the base model with updated weights parameter
base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
# base_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
# base_model = models.efficientnet_b7(pretrained=True)  # EfficientNet B7 is one of the most powerful models



# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augment_transform=None, class_threshold=15):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform
        self.class_threshold = class_threshold

        # Count how many images exist per class
        self.class_counts = self.labels.iloc[:, 0].value_counts().to_dict()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 0]

        # If the class has fewer than 'class_threshold' images, apply augmentation
        if self.class_counts[label] < self.class_threshold:
            if self.augment_transform and torch.rand(1).item() > 0.5:
                image = self.augment_transform(image)
            else:
                image = self.transform(image)
        else:
            if self.transform:
                image = self.transform(image)

        return image, label

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# Define transformations
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with resizing
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])

# Split data into training and validation
labels_df = pd.read_csv('data/train.csv')
train_labels, val_labels = train_test_split(labels_df, test_size=0.1, stratify=labels_df.iloc[:, 0], random_state=42)
train_labels.to_csv('data/train_split.csv', index=False)
val_labels.to_csv('data/val_split.csv', index=False)

# Create datasets and DataLoaders
data_dir = 'data/train'
train_dataset = ImageDataset(csv_file='data/train_split.csv', root_dir=data_dir, transform=base_transform, augment_transform=augment_transform)
val_dataset = ImageDataset(csv_file='data/val_split.csv', root_dir=data_dir, transform=base_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Freeze all layers initially
for param in base_model.parameters():
    param.requires_grad = False

# Unfreeze the last block (layer4) for fine-tuning
for param in base_model.layer4.parameters():  
    param.requires_grad = True

# Modify the fully connected (fc) layer to match the number of classes (488)
base_model.fc = nn.Linear(base_model.fc.in_features, 488)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.fc.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# Training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
base_model.to(device)

num_epochs = 30
best_val_logloss = float('inf')
best_val_acc = 0.0
patience_no_imprv = 4
epochs_no_imprv = 0

for epoch in range(num_epochs):
    base_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation loop
    base_model.eval()
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model outputs
            outputs = base_model(inputs)
            
            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)  # Shape: (batch_size, 488)
            _, preds = torch.max(probs, 1)  # Predicted class labels
            
            # Convert tensors to numpy arrays and accumulate
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert lists to numpy arrays for log_loss and accuracy
    all_labels = np.array(all_labels)  # Shape: (num_samples, )
    all_probs = np.array(all_probs)  # Shape: (num_samples, 488)

    # Calculate average log loss and accuracy
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(488)))
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Log Loss: {val_logloss:.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_logloss < best_val_logloss:  # Lower log loss is better
        best_val_logloss = val_logloss
        best_val_acc = val_acc
        torch.save(base_model.state_dict(), 'models/resnet/base_model_fined.pth')
        epochs_no_imprv = 0
    else:
        epochs_no_imprv += 1

    if epochs_no_imprv >= patience_no_imprv:
        print("Early stopping triggered")
        break

    lr_scheduler.step(val_logloss)

print(f"Training completed. Best validation log loss: {best_val_logloss:.4f}. Best validation accuracy: {best_val_acc:.4f} ")
