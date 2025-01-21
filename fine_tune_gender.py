import os
import timm
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()



#######################################################

model_name = 'vit_l_16'
batch_size = 8
num_epochs = 80
patience_no_imprv = 12
test_size = 0.1
learning_rate = 0.001


# Load the base model with updated weights parameter
match model_name:
    case 'resnet101':
        base_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    case 'resnet152':
        base_model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    case 'vit_b_16':
        base_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    case 'vit_l_16':
        base_model = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
    case 'efficientnet_b7_ns':
        base_model = timm.create_model('tf_efficientnet_b7.ns_jft_in1k', pretrained=True, num_classes=2)



#######################################################

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

        if self.augment_transform and torch.rand(1).item() > 0.3:
            image = self.augment_transform(image)
        else:
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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop with resizing
    transforms.RandAugment(num_ops=3, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(*imagenet_stats)
])



#######################################################

# Split data into training and validation
labels_df = pd.read_csv('data/bodies/bodies_labels.csv')
train_labels, val_labels = train_test_split(labels_df, test_size=test_size, stratify=labels_df.iloc[:, 0], random_state=42)
train_labels.to_csv('data/bodies/train_split.csv', index=False)
val_labels.to_csv('data/bodies/val_split.csv', index=False)

# Calculate the class weights
labels_df_train = pd.read_csv('data/bodies/train_split.csv')
class_counts = labels_df_train.iloc[:, 0].value_counts()  # Count number of images per class
total_samples = len(labels_df_train)
# Inverse frequency weighting: more images = smaller weight, fewer images = larger weight
class_weights = {label: total_samples / (len(class_counts) * count) for label, count in class_counts.items()}
# Convert the class weights to a tensor for PyTorch
weights = torch.tensor([class_weights[label] for label in range(2)], dtype=torch.float32).to(device)

# Create datasets and DataLoaders
data_dir = 'data/bodies'
train_dataset = ImageDataset(csv_file='data/bodies/train_split.csv', root_dir=data_dir, transform=base_transform, augment_transform=augment_transform)
val_dataset = ImageDataset(csv_file='data/bodies/val_split.csv', root_dir=data_dir, transform=base_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



#######################################################

match model_name:

    case 'resnet101' | 'resnet152':
        # Freeze all layers initially
        for param in base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last block (layer4) for fine-tuning
        for param in base_model.layer4.parameters():  
            param.requires_grad = True

        # Modify the fully connected (fc) layer to match the number of classes (2)
        base_model.fc = nn.Linear(base_model.fc.in_features, 2)
        optimizer_parameters = base_model.parameters()

    case 'vit_b_16' | 'vit_l_16':
        # Freeze early layers in the encoder
        for name, param in base_model.named_parameters():
            # Check if the layer belongs to the encoder and is within the first few layers
            if "encoder" in name and "layer" in name.split('.'):
                param.requires_grad = True

        # Modify the head layer to match the number of classes (2)
        base_model.heads = nn.Linear(base_model.heads[0].in_features, 2)
        optimizer_parameters = base_model.heads.parameters()

    case 'efficientnet_b7_ns':
        for param in base_model.parameters():
            param.requires_grad = True
        optimizer_parameters = base_model.parameters()


torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate)
# optimizer = torch.optim.RMSprop(optimizer_parameters, lr=0.001, momentum=0.9, weight_decay=0.00001, alpha=0.9)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)



#######################################################

# Training the model
base_model.to(device)

best_train_loss = float('inf')
best_avg_val_loss = float('inf')
best_val_logloss = float('inf')
best_val_acc = 0.0
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
    val_loss = 0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get model outputs
            outputs = base_model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Apply softmax to convert logits to probabilities
            probs = torch.softmax(outputs, dim=1)  # Shape: (batch_size, 2)
            _, preds = torch.max(probs, 1)  # Predicted class labels
            
            # Convert tensors to numpy arrays and accumulate
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert lists to numpy arrays for log_loss and accuracy
    all_labels = np.array(all_labels)  # Shape: (num_samples, )
    all_probs = np.array(all_probs)  # Shape: (num_samples, 2)

    # Calculate parameters
    train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    val_logloss = log_loss(all_labels, all_probs, labels=list(range(2)))
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Log Loss: {val_logloss:.4f}, Validation Accuracy: {val_acc:.4f}")

    if val_logloss < best_val_logloss or (val_logloss == best_val_logloss and train_loss < best_train_loss):
        best_train_loss = train_loss
        best_avg_val_loss = avg_val_loss
        best_val_logloss = val_logloss
        best_val_acc = val_acc
        torch.save(base_model.state_dict(), f'models/gender/{model_name}_gender_fined.pth')
        epochs_no_imprv = 0
    else:
        epochs_no_imprv += 1

    if epochs_no_imprv >= patience_no_imprv:
        print("Early stopping triggered")
        break

    lr_scheduler.step(val_logloss)

print(f"Training completed. Best Training Loss: {best_train_loss:.4f}. Best validation loss: {best_avg_val_loss:.4f}. Best validation log loss: {best_val_logloss:.4f}. Best validation accuracy: {best_val_acc:.4f} ")
