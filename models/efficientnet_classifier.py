# pip install efficientnet-pytorch
from efficientnet_pytorch import EfficientNet
from models.base_classifier import BaseClassifier
import torch.nn as nn
import torch

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
    
    @classmethod
    def load_model(cls, model_weight_path, **kwargs):
        """
        Creates an instance of the model and loads the weights from a checkpoint.
        
        Args:
          model_weight_path (str): The file path to the saved weights.
          **kwargs: All other keyword args required to instantiate the model (e.g., num_classes,
                    train_path, etc.).
                    
        Returns:
          An instance of Efficientnet in evaluation mode.
        """
        # Instantiate the model with provided kwargs
        model = cls(**kwargs)
        
        # Load the saved state dictionary
        state_dict = torch.load(model_weight_path, map_location="cpu")
        
        # Optionally adjust keys if the file was saved without the "efficientnet_model." prefix.
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("efficientnet_model."):
            state_dict = {"efficientnet_model." + key: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set the model to evaluation mode
        return model