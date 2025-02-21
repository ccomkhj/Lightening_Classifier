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
   - Supports EfficientNet-B0 to B7.
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

After completion, you will get report in csv as below.
Based on the metric, decide which model is appropriate for your task.
```
| Model            | Test Accuracy   | Test Precision  | Test Recall     | Test F1         | Training Time  | Timestamp      |
|------------------|-----------------|-----------------|-----------------|-----------------|----------------|----------------|
| ResNext101       | 0.78899085521698| 0.7996436357498169| 0.78899085521698| 0.7887938022613525| 1:47:51.260405 | 20250221_094927|
| ResNet101        | 0.78899085521698| 0.7946293354034424| 0.78899085521698| 0.7877031564712524| 1:03:14.367643 | 20250221_113730|
| SwinTransformer  | 0.8807339668273926| 0.8827170133590698| 0.8807339072227478| 0.8803048133850098| 0:22:18.137873 | 20250221_124052|
| ViT              | 0.8440366983413696| 0.8472263813018799| 0.8440366983413696| 0.8441518545150757| 0:22:27.548590 | 20250221_130314|
| DenseNet121      | 0.853210985660553| 0.8623536229133606| 0.853210985660553| 0.8529086112976074| 0:58:01.608439 | 20250221_132545|
| EfficientNetB7   | 0.8440366983413696| 0.8501067161560059| 0.8440366983413696| 0.844648003578186| 3:06:57.563459 | 20250221_142354|
```

Use the `demo.ipynb` to follow the whole workflow with a model.


## Extending the Repository:
**To add a new model**:
- Create a new Python file under `models` directory. (e.g., new_model_classifier.py).
- Inherit from BaseClassifier and implement the model-specific logic.
- If needed, Add the new model to the `models_to_test` dictionary in [train_all_models.py](train_all_models.py).

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.