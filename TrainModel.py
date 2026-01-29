"""
This script trains an EfficientNet-B0 model for binary classification
(e.g., Diabetic Retinopathy detection) using transfer learning.

It performs training in two phases: initial training with the feature extractor frozen,
followed by fine-tuning with partial unfreezing of the last few layers.

The script uses provided training and validation directories, applies custom data transformations,
and saves the trained model with a versioned filename.

Usage:
    python TrainModel.py <train_dir> <test_dir>

Note:
    The train and test (validation) directories must be different.
    Hyperparameters such as image size, batch size, epochs, learning rates,
    and early stopping accuracy can be adjusted within the script.
"""
import torch
import ProjectModules as modules
from torch import nn, optim
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python TrainModel.py <train_dir> <test_dir>")
        exit(1)
    elif sys.argv[1] == sys.argv[2]:
        print("Train and test dirs cannot be the same")
        exit(1)

    # Definitions (these can be tweaked, these are the defaults)
    IMG_SIZE = 227
    BATCH_SIZE = 8
    PHASE1_EPOCHS = 10
    PHASE2_EPOCHS = 15
    PHASE1_LR = 1e-3
    PHASE2_LR = 1e-5
    VERSION = 4
    EARLY_STOPPING_ACC = 1 # if this is 1, early stopping does not occur. This value is compared to the validation accuracy

    train_transform = modules.get_custom_transform(IMG_SIZE, True)
    val_transform = modules.get_custom_transform(IMG_SIZE, False)

    train_loader = modules.get_data_loader(sys.argv[1], train_transform, BATCH_SIZE)  # Get data loader objects
    val_loader = modules.get_data_loader(sys.argv[2], val_transform, BATCH_SIZE, shuffle=False)

    device = modules.get_device() # Get computing device object

    model = modules.get_model(weights="IMAGENET1K_V1") # Get model object

    modules.freeze_feature_extractor(model) # Freeze feature extractor

    num_filters = modules.get_num_filters(model) # Replace classifier for binary classification

    model.classifier = modules.get_classifier(num_filters) # Replace last layer

    model = model.to(device) # Move model to device wrapper object

    # Loss and optimizer objects
    criterion = nn.BCEWithLogitsLoss()   # for binary + single neuron output
    optimizer = optim.Adam(model.classifier.parameters(), lr=PHASE1_LR)

    # Phase 1 training (frozen)
    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = modules.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = modules.validate(model, device, val_loader, criterion)

        print(f"Epoch {epoch+1}/{PHASE1_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}\n")

    # unfreeze last few blocks
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=PHASE2_LR) # lower LR for stable-fine tuning

    # Train again
    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = modules.train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = modules.validate(model, device, val_loader, criterion)

        print(f"[FT] Epoch {epoch+1}/{PHASE2_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}  Val Acc:   {val_acc:.4f}\n")

        if val_acc >= EARLY_STOPPING_ACC: # Early stopping, good enough, stop before it gets worse
            break

    print(f"Training complete! Saving model version {VERSION}...")
    print(f"Phase 1 Learning Rate: {PHASE1_LR:.2e}  Phase 2 Learning Rate: {PHASE2_LR:.2e}")
    print(f"Phase 1 Epochs: {PHASE1_EPOCHS}  Phase 2 Epochs: {PHASE2_EPOCHS}")

    torch.save(model.state_dict(), f"efficientnet_b0_binaryV{VERSION}_.pt") # Save model
