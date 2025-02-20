# PyTorch Lightning Image Classifiers

This repository contains modular PyTorch Lightning implementations of popular deep learning models for image classification. All models inherit from a common `BaseClassifier` class, making it easy to modify, extend, and use for various tasks.

## Models Included

1. **ResNetClassifier** (`resnet_classifier.py`)
   - Supports ResNet versions: 18, 34, 50, 101, 152.
   - Configurable for transfer learning and fine-tuning.

2. **ResNextClassifier** (`resnext_classifier.py`)
   - Supports ResNext versions: 50, 101.
   - Designed for high-performance classification.

3. **DenseNetClassifier** (`densenet_classifier.py`)
   - Supports DenseNet-121.
   - Efficient feature reuse with dense connections.

4. **EfficientNetClassifier** (`efficientnet_classifier.py`)
   - Supports EfficientNet-B0.
   - Lightweight and scalable for small datasets.

5. **ViTClassifier** (`vit_classifier.py`)
   - Vision Transformer (ViT) for global context modeling.
   - Pre-trained on ImageNet.

6. **SwinTransformerClassifier** (`swin_transformer_classifier.py`)
   - Swin Transformer for hierarchical feature extraction.
   - Pre-trained on ImageNet.

## Features

- **Modular Design**: All models inherit from `BaseClassifier`, ensuring consistent training, validation, and testing workflows.
- **Easy Configuration**: Modify hyperparameters like learning rate, batch size, and optimizer directly in the configuration.
- **Checkpointing**: Automatically saves the best model during training.
- **Early Stopping**: Prevents overfitting by stopping training if validation performance plateaus.

## Usage

1. **Install Dependencies**:
```bash
pip install pytorch_lightning torchvision transformers efficientnet-pytorch
```

2. **Train and Test Models**:
Use the `train_all_models.py` script to train and test all models and check what works the best.:


## Extending the Repository**:
**To add a new model**:
- Create a new Python file (e.g., new_model_classifier.py).
- Inherit from BaseClassifier and implement the model-specific logic.
- Add the new model to the models_to_test dictionary in train_all_models.py.

## License
This project is licensed under the MIT License. See LICENSE for details.