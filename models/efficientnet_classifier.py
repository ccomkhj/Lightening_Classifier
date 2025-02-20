# pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
from models.base_classifier import BaseClassifier
import torch.nn as nn

class EfficientNetClassifier(BaseClassifier):
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
        target_size=(730, 968), 
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
        
        # Load pre-trained EfficientNet
        self.efficientnet_model = EfficientNet.from_pretrained('efficientnet-b7') #b1-b7
        
        # Replace the classifier head
        linear_size = self.efficientnet_model._fc.in_features
        self.efficientnet_model._fc = nn.Linear(linear_size, num_classes)
        
        # Freeze layers if needed
        if tune_fc_only:
            for param in self.efficientnet_model.parameters():
                param.requires_grad = False
            for param in self.efficientnet_model._fc.parameters():
                param.requires_grad = True

    def forward(self, X):
        return self.efficientnet_model(X)