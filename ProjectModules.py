import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

"""
This module provides utility functions for data loading, model configuration, 
and training/validation routines used in an EfficientNet-B0 based binary 
classification project (e.g., Diabetic Retinopathy detection).
"""

# Data augmentation and normalization for our dataset

def get_custom_transform(IMG_SIZE, isTrain):
    """ Returns a custom transform for the dataset based on the isTrain flag
        Returns a Compose object of transforms, one for train and one for test.
    """
    if isTrain:
        return transforms.Compose([ # custom transforms specific to this project
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0) ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225] )
            ])
    else:
        return transforms.Compose([
            transforms.Resize( (IMG_SIZE, IMG_SIZE) ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225] )
        ])

def get_data_loader(data_dir: str, transform, batch_size, shuffle=True):
    """Load the datasets with dir, transform and batch size."""
    ds = datasets.ImageFolder(data_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def get_device():
    """Returns the computing device wrapper object"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(weights=None):
    """Returns effnet b0 (pre trained)"""
    return models.efficientnet_b0(weights=weights)

def get_num_filters(model):
    """
    Returns the number of input features for the classifier layer of the given model.
    """
    return model.classifier[1].in_features

def get_classifier(num_filters):
    """
    Returns a custom Sequential classifier for binary classification with dropout and a single output neuron.
    """
    return nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_filters, 1)  # 1 output -> sigmoid later
    )

def freeze_feature_extractor(model): # Freeze feature extractor
    """
    Freezes all parameters in the feature extractor layers of the model to prevent updates during training.
    """
    for param in model.features.parameters():
        param.requires_grad = False

# Training and validation loops
def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using the provided loader, optimizer, and criterion.
    
    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).int()
        correct += (preds == labels.int()).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, device, loader, criterion):
    """
    Validates the model on the provided loader using the given criterion.
    
    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total
