# EfficientNet-B0 Binary Classification

A PyTorch implementation of binary image classification using EfficientNet-B0 with transfer learning. Designed for medical image classification tasks such as Diabetic Retinopathy detection.

## Features

- **Two-phase training**: Initial frozen feature extractor training followed by fine-tuning
- **Custom data augmentation**: Optimized transformations for medical imaging
- **Transfer learning**: Uses ImageNet pre-trained EfficientNet-B0
- **Early stopping**: Prevents overfitting with configurable accuracy threshold
- **Modular design**: Reusable components in `ProjectModules.py`

## Requirements

```bash
pip install torch torchvision tqdm
```

## Usage

```bash
python TrainModel.py <train_dir> <test_dir>
```

### Directory Structure

Your training and validation directories should follow this structure:

```
train_dir/
  ├── class_0/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── class_1/
      ├── image3.jpg
      └── image4.jpg
```

## Configuration

Hyperparameters can be adjusted in `TrainModel.py`:

- `IMG_SIZE`: Input image size (default: 227)
- `BATCH_SIZE`: Batch size (default: 8)
- `PHASE1_EPOCHS`: Frozen training epochs (default: 10)
- `PHASE2_EPOCHS`: Fine-tuning epochs (default: 15)
- `PHASE1_LR`: Initial learning rate (default: 1e-3)
- `PHASE2_LR`: Fine-tuning learning rate (default: 1e-5)
- `EARLY_STOPPING_ACC`: Validation accuracy threshold (default: 1.0)

## Training Process

### Phase 1: Frozen Feature Extractor
- Trains only the classifier head
- Higher learning rate (1e-3)
- Prevents catastrophic forgetting

### Phase 2: Fine-Tuning
- Unfreezes last 3 blocks of feature extractor
- Lower learning rate (1e-5)
- Adapts features to target domain

## Model Output

The trained model is saved as `efficientnet_b0_binaryV{VERSION}_.pt` and can be loaded using:

```python
import torch
from torchvision import models

model = models.efficientnet_b0()
# Replace classifier to match training configuration
model.load_state_dict(torch.load('efficientnet_b0_binaryV4_.pt'))
```

## Module Documentation

See `ProjectModules.py` for detailed documentation of utility functions including:
- Data loading and transformation
- Model configuration
- Training and validation loops
