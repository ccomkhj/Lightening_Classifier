# pip install transformers
from transformers import ViTForImageClassification
from models.base_classifier import BaseClassifier
import torch.nn as nn
import torch

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
    
    @classmethod
    def load_model(cls, model_weight_path, **kwargs):
        """
        Creates an instance of the model and loads the weights from a checkpoint.
        
        Args:
          model_weight_path (str): The file path to the saved weights.
          **kwargs: All other keyword args required to instantiate the model (e.g., num_classes,
                    train_path, etc.).
                    
        Returns:
          An instance of ViTClassifier in evaluation mode.
        """
        # Instantiate the model with provided kwargs
        model = cls(**kwargs)
        
        # Load the saved state dictionary
        state_dict = torch.load(model_weight_path, map_location="cpu")
        
        # Optionally adjust keys if the file was saved without the "vit_model." prefix.
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("vit_model."):
            state_dict = {"vit_model." + key: value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # Set the model to evaluation mode
        return model