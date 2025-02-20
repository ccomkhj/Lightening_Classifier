from models.resnet_classifier import ResNetClassifier
from models.swin_transformer_classifier import SwinTransformerClassifier
from models.vit_classifier import ViTClassifier
from models.densenet_classifier import DenseNetClassifier
from models.efficientnet_classifier import EfficientNetClassifier

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
import os
import csv
from pathlib import Path
from datetime import datetime

# Configuration
DATA_ROOT = Path("split")  # Update this path
train_path = DATA_ROOT / "train"
val_path = DATA_ROOT / "val"
test_path = DATA_ROOT / "test"

# Common model configuration
model_config = {
    "num_classes": 3,
    "train_path": train_path,
    "val_path": val_path,
    "test_path": test_path,
    "optimizer": "adam",
    "lr": 1e-3,
    "batch_size": 32,
    "transfer": True,
    "tune_fc_only": True,
}

# Model-specific configurations
models_to_test = {
    "ResNet101": {
        "class": ResNetClassifier,
        "config": {**model_config, "resnet_version": 101, "target_size": (730, 968)},
    },
    "SwinTransformer": {
        "class": SwinTransformerClassifier,
        "config": {**model_config, "target_size": (384, 384)},
    },
    "ViT": {
        "class": ViTClassifier,
        "config": {**model_config, "target_size": (384, 384)},
    },
    "DenseNet121": {
        "class": DenseNetClassifier,
        "config": {**model_config, "target_size": (730, 968)},
    },
    "EfficientNetB0": {
        "class": EfficientNetClassifier,
        "config": {**model_config, "target_size": (730, 968)},
    },
}

# Create a timestamp for the log file
now = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"model_performance_log_{now}.csv"  # Include timestamp in the log file name
log_columns = ["Model", "Test Accuracy", 'Test Precision', 'Test Recall', 'Test F1', "Training Time", "Timestamp"]

# Check if the log file exists, if not, create it and write the header
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_columns)
        writer.writeheader()

# Training and testing loop
for model_name, model_info in models_to_test.items():
    print(f"Training and testing {model_name}...")

    # Initialize model
    model_class = model_info["class"]
    config = model_info["config"]
    model = model_class(**config)

    # Create a timestamp for this model's training
    model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = model_name.lower()

    # Training setup
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{model_timestamp}",  # Save checkpoints inside "checkpoints/{timestamp}" folder
        filename=f"{arch}_epoch{{epoch}}.pth",  # Filename includes model architecture and epoch number
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        patience=60,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=120,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train the model
    start_time = datetime.now()
    trainer.fit(model)
    training_time = datetime.now() - start_time

    # Test the model (if test set available)
    test_accuracy = None
    if test_path.exists():
        test_results = trainer.test(ckpt_path="best")
        test_accuracy = test_results[0]["test_acc"]
        test_precision = test_results[0]["test_precision"]
        test_recall = test_results[0]["test_recall"]
        test_f1 = test_results[0]["test_f1"]

    # Log metrics to the log file
    with open(log_file, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_columns)
        writer.writerow({
            "Model": model_name,
            "Test Accuracy": test_accuracy,
            "Test Precision": test_precision,
            "Test Recall": test_recall,
            "Test F1": test_f1,
            "Training Time": str(training_time),
            "Timestamp": model_timestamp,
        })

    # Save final model weights
    os.makedirs("saved_model", exist_ok=True)
    final_filename = f"saved_model/{arch}_final_{model_timestamp}.pth"
    if hasattr(model, "resnet_model"):  # For ResNet
        torch.save(model.resnet_model.state_dict(), final_filename)
    elif hasattr(model, "swin_model"):  # For Swin Transformer
        torch.save(model.swin_model.state_dict(), final_filename)
    elif hasattr(model, "vit_model"):  # For ViT
        torch.save(model.vit_model.state_dict(), final_filename)
    elif hasattr(model, "densenet_model"):  # For DenseNet
        torch.save(model.densenet_model.state_dict(), final_filename)
    elif hasattr(model, "efficientnet_model"):  # For EfficientNet
        torch.save(model.efficientnet_model.state_dict(), final_filename)

    print(f"Finished training and testing {model_name}.\n")
