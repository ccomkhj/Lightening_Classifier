from models.resnet_classifier import ResNetClassifier
from models.swin_transformer_classifier import SwinTransformerClassifier
from models.vit_classifier import ViTClassifier
from models.densenet_classifier import DenseNetClassifier
from models.efficientnet_classifier import EfficientNetClassifier

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
import os
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
        "config": {**model_config, "target_size": (224, 224)},
    },
    "ViT": {
        "class": ViTClassifier,
        "config": {**model_config, "target_size": (224, 224)},
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

# Training and testing loop
for model_name, model_info in models_to_test.items():
    print(f"Training and testing {model_name}...")

    # Initialize model
    model_class = model_info["class"]
    config = model_info["config"]
    model = model_class(**config)

    # Create a timestamp and get architecture name
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    arch = model_name.lower()

    # Training setup
    checkpoint_cb = ModelCheckpoint(
        dirpath=f"checkpoints/{now}",  # Save checkpoints inside "checkpoints/{now}" folder
        filename=f"{arch}_epoch{{epoch}}.pth",  # Filename includes model architecture and epoch number
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        patience=20,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=2,
        callbacks=[checkpoint_cb, early_stop_cb],
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(model)

    # Test the model (if test set available)
    if test_path.exists():
        trainer.test(ckpt_path="best")
    
    os.makedirs("saved_model", exist_ok=True)

    # Construct dynamic filename for the final model
    final_filename = f"saved_model/{arch}_final_{now}.pth"

    # Save final model weights with the dynamic filename
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