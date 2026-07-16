import torch
import torch.nn as nn

class LightweightMambaHead(nn.Module):
    def __init__(self, input_channels, d_model=128, num_classes=101):
        """
        Args:
            input_channels (int): The 'C' dimension from your CNN output (e.g., 40).
            d_model (int): The hidden dimension for Mamba (keep it low for Jetson, e.g., 128 or 256).
            num_classes (int): Number of action classes (101 for UCF101 dataset).
        """
        super().__init__()
        
        # Spatial Pooling
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to hidden dimension
        self.proj = nn.Linear(input_channels, d_model)
        
        # The Mamba Temporal Modeler
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,  # Standard SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2     # Block expansion factor
        )
        
        self.dropout = nn.Dropout(p=0.5)

        # Final Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        Expects input from CNN: (Batch, Frames, Channels, Height, Width)
        """
        B, F, C, H, W = x.shape
        
        # Reshape to combine Batch and Frames so we can apply 2D pooling
        x_flat = x.view(B * F, C, H, W)
        
        # Pool spatial dimensions
        pooled = self.spatial_pool(x_flat)
        pooled = pooled.view(B * F, C)
        
        # Separate Batch and Frames back
        sequence_tokens = pooled.view(B, F, C)
        
        # Project channels to d_model
        sequence_tokens = self.proj(sequence_tokens)
        
        # Pass the sequence through Mamba
        mamba_out = self.mamba(sequence_tokens)
        
        # Temporal Pooling to get a single vector for whole video
        # (B, F, d_model) -> (B, d_model)
        video_feature = mamba_out.mean(dim=1)

        video_feature = self.dropout(video_feature)
        
        # Final classification
        logits = self.classifier(video_feature)
        
        return logits