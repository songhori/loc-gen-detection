import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Custom dataset class
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, augment_transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 1])
        image = Image.open(img_name).convert('RGB')
        label = self.labels.iloc[idx, 0]

        if self.augment_transform and torch.rand(1).item() > 0.5:
            image = self.augment_transform(image)
        elif self.transform:
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

# Load the ResNet101 model with updated weights parameter
resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

# Freeze all layers initially
for param in resnet101.parameters():
    param.requires_grad = False

# Unfreeze the last block (layer4) for fine-tuning
for param in resnet101.layer4.parameters():  
    param.requires_grad = True

# Modify the fully connected (fc) layer to match the number of classes (488)
resnet101.fc = nn.Linear(resnet101.fc.in_features, 488)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet101.fc.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet101.to(device)

num_epochs = 15
best_val_acc = 0.0

for epoch in range(num_epochs):
    resnet101.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = resnet101(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    resnet101.eval()
    val_preds, val_labels_list = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = resnet101(inputs)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_list.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels_list, val_preds)
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(resnet101.state_dict(), 'resnet101_fined.pth')

    lr_scheduler.step()

print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
