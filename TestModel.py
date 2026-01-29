"""
This script tests a pre-trained EfficientNet-B0 model on a test dataset for binary classification
(e.g., Diabetic Retinopathy detection). It loads the specified saved model, performs inference on
images from the given test directory, computes performance metrics including accuracy, sensitivity,
and specificity, and displays a confusion matrix.

Usage:
    python TestModel.py <test_dir> <saved_model_name>

Note:
    The saved model must be in the same directory as this script.
    The test directory should contain subfolders for each class (e.g., 'DR' and 'NonDR').
"""
from torch.utils.data import DataLoader
import ProjectModules as modules
import torch
from torchvision import datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python TestModel.py <test_dir> <saved_model_name>\n Note: saved model must be in same directory as script")
        exit(1)

    # Definitions
    DIR = sys.argv[1]
    VERSION = sys.argv[2]
    IMG_SIZE = 227
    BATCH_SIZE = 32

    print("Loading data and model...")
    transform = modules.get_custom_transform(IMG_SIZE, isTrain=False) # Get custom transform settings object

    test_ds = datasets.ImageFolder(DIR, transform=transform) # Get test dataset, pass transform settings

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False) # Get data loader object

    device = modules.get_device() # Get computing device object

    model = modules.get_model() # Get model object

    num_filters = modules.get_num_filters(model)
                                                  # Replace classifier for binary classification
    model.classifier = modules.get_classifier(num_filters)

    model.load_state_dict(torch.load(VERSION, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode

    all_preds = []
    all_labels = []

    print("Done! \nStarting test...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Test Complete!")

    cm = confusion_matrix(all_labels, all_preds) # Make confusion matrix obj

    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")

    disp = ConfusionMatrixDisplay( # Create confusion matrix wrapper object
        confusion_matrix=cm,
        display_labels=test_ds.classes  # ["DR", "NonDR"]
    )
    disp.plot(values_format='d')
    plt.title("EfficientNet-B0 Test Confusion Matrix")
    plt.show()
