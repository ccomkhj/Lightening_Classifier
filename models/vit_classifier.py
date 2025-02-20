# pip install transformers
from transformers import ViTForImageClassification
from models.base_classifier import BaseClassifier
import torch.nn as nn

class ViTClassifier(BaseClassifier):
    def __init__(
        self,
        num_classes,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
        target_size=(384, 384),
    ):
        super().__init__(
            num_classes=num_classes,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            transfer=transfer,
            tune_fc_only=tune_fc_only,
            target_size=target_size,
        )
        
        # Load pre-trained Vision Transformer (ViT)
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
        
        """
        google/vit-base-patch16-224 (224x224)
        google/vit-base-patch16-384 (384x384)
        google/vit-large-patch16-224 (224x224)
        google/vit-large-patch16-384 (384x384)
        """
        
        # Replace the classifier head
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, num_classes)
        
        # Freeze layers if needed
        if tune_fc_only:
            for param in self.vit_model.parameters():
                param.requires_grad = False
            for param in self.vit_model.classifier.parameters():
                param.requires_grad = True

    def forward(self, X):
        # Resize input to target size if necessary
        if X.shape[-2:] != self.target_size:
            X = nn.functional.interpolate(X, size=self.target_size, mode="bilinear", align_corners=False)
        return self.vit_model(X).logits