import torch
import torch.nn as nn

from models.cnn_extractor import TruncatedMobileNetV3
from models.mamba_block import LightweightMambaHead

class CNNMambaEdge(nn.Module):
    def __init__(self, num_classes=101, handoff_index=5, cnn_size='small', mamba_dim=128):
        """
        The Master Hybrid Model for Jetson Orin Nano
        """
        super().__init__()
        print(f"Initializing Model (Handoff Depth: {handoff_index})")
        
        # Initialize CNN block
        self.cnn = TruncatedMobileNetV3(
            handoff_index=handoff_index, 
            model_size=cnn_size, 
            freeze_weights=True # Keep frozen for faster training
        )
        
        # Dynamically calculate the CNN output channels
        # We pass a tiny 1-frame fake video to see what the CNN spits out at this depth
        dummy_video = torch.randn(1, 1, 3, 224, 224)
        with torch.no_grad():
            dummy_features = self.cnn(dummy_video)
            
        cnn_out_channels = dummy_features.shape[2]
        print(f"CNN Output Channels at depth {handoff_index}: {cnn_out_channels}")
        
        # 3. Initialize the Mamba Temporal Modeler
        self.mamba = LightweightMambaHead(
            input_channels=cnn_out_channels, 
            d_model=mamba_dim, 
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Input: (Batch, Frames, Channels, Height, Width)
        Output: (Batch, Classes)
        """
        # Extract spatial features for each frame
        spatial_features = self.cnn(x)
        
        # Process the temporal sequence and classify
        logits = self.mamba(spatial_features)
        
        return logits