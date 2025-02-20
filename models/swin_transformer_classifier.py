# pip install transformers
from transformers import SwinForImageClassification
from models.base_classifier import BaseClassifier
import torch.nn as nn

class SwinTransformerClassifier(BaseClassifier):
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
        target_size=(224, 224),
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
        
        # Load pre-trained Swin Transformer
        self.swin_model = SwinForImageClassification.from_pretrained('microsoft/swin-tiny-patch4-window7-224')
        
        # Replace the classifier head
        self.swin_model.classifier = nn.Linear(self.swin_model.config.hidden_size, num_classes)
        
        # Freeze layers if needed
        if tune_fc_only:
            for param in self.swin_model.parameters():
                param.requires_grad = False
            for param in self.swin_model.classifier.parameters():
                param.requires_grad = True

    def forward(self, X):
        # Resize input to (224, 224) if necessary
        if X.shape[-2:] != (224, 224):
            X = torch.nn.functional.interpolate(X, size=(224, 224), mode="bilinear", align_corners=False)
        return self.swin_model(X).logits