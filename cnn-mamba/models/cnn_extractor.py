import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

class TruncatedMobileNetV3(nn.Module):
    def __init__(self, handoff_index, model_size='small', freeze_weights=True):
        """
        Args:
            handoff_index (int): How many layers deep to go before passing to Mamba.
                                 MobileNetV3-Small has 13 feature blocks (0 to 12).
            model_size (str): 'small' or 'large'. 'small' is recommended for Jetson.
            freeze_weights (bool): Freezes CNN weights to speed up early training.
        """
        super().__init__()
        
        # Load the pre-trained model
        if model_size == 'small':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            base_model = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
            base_model = models.mobilenet_v3_large(weights=weights)
            
        # Slice the 'features' sequential block up to the handoff_index
        self.spatial_extractor = nn.Sequential(
            *list(base_model.features.children())[:handoff_index]
        )
        
        # Freeze the weights
        if freeze_weights:
            for param in self.spatial_extractor.parameters():
                param.requires_grad = False
                
    def forward(self, x):
        """
        Expects video input shape: (Batch, Frames, Channels, Height, Width)
        """
        B, F, C, H, W = x.shape
        
        # Shape becomes: (Batch * Frames, Channels, Height, Width)
        x_reshaped = rearrange(x, 'b f c h w -> (b f) c h w')
        
        spatial_features = self.spatial_extractor(x_reshaped)
        _, out_c, out_h, out_w = spatial_features.shape

        # Shape becomes: (Batch, Frames, Out_Channels, Out_Height, Out_Width)
        features_separated = rearrange(
            spatial_features, 
            '(b f) c h w -> b f c h w', 
            b=B, 
            f=F
        )
        
        return features_separated