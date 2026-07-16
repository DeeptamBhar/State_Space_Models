import os
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision import transforms

class UCF101Dataset(Dataset):
    def __init__(self, data_dir, split_file, num_frames=16, is_training=True):
        self.data_dir = data_dir
        self.num_frames = num_frames
        self.is_training = is_training
        
        self.video_paths = []
        self.labels = []
        
        # NEW: Load the master class index to map strings to integers
        class_ind_file = os.path.join(os.path.dirname(split_file), 'classInd.txt')
        class_to_idx = {}
        with open(class_ind_file, 'r') as f:
            for line in f:
                idx, class_name = line.strip().split()
                class_to_idx[class_name] = int(idx) - 1 # PyTorch needs 0-indexed labels
                
        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                self.video_paths.append(parts[0])
                
                # If it's the train set, the label is in parts[1]
                if len(parts) > 1:
                    self.labels.append(int(parts[1]) - 1)
                # If it's the val set, we extract the class name from the folder path
                else:
                    class_name = parts[0].split('/')[0]
                    self.labels.append(class_to_idx[class_name]) 

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        vid_path = os.path.join(self.data_dir, self.video_paths[idx])
        label = self.labels[idx]
        
        try:
            # Use Decord to read the video efficiently
            vr = VideoReader(vid_path, ctx=cpu(0))
            total_frames = len(vr)
            
            # Sample exactly 'num_frames' evenly across the video length
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            # Load the frames (returns a NumPy array of shape: Frames, Height, Width, Channels)
            frames = vr.get_batch(indices).asnumpy()
        except Exception as e:
            print(f"Error loading {vid_path}: {e}")
            # Fallback: Return a zero tensor if a video is corrupted
            return torch.zeros((self.num_frames, 3, 224, 224)), label

        # Process each frame through our transforms
        processed_frames = []
        for frame in frames:
            # frame is currently (H, W, C) numpy array.
            # transform converts to (C, H, W) torch tensor.
            tensor_frame = self.transform(frame)
            processed_frames.append(tensor_frame)
            
        # Stack frames back together into (Frames, Channels, Height, Width)
        video_tensor = torch.stack(processed_frames)
        
        return video_tensor, torch.tensor(label, dtype=torch.long)

# ==========================================
# Phase 2: Dataloader Test
# ==========================================
if __name__ == "__main__":
    # Adjust these paths if your folders are named differently!
    DATA_DIR = "data/raw/UCF-101"
    SPLIT_FILE = "data/raw/ucfTrainTestlist/trainlist01.txt"
    
    # Check if data exists before running
    if os.path.exists(DATA_DIR) and os.path.exists(SPLIT_FILE):
        print("Dataset found! Initializing DataLoader...")
        dataset = UCF101Dataset(data_dir=DATA_DIR, split_file=SPLIT_FILE, num_frames=16)
        
        # Grab the first video in the dataset
        video, label = dataset[0]
        
        print("-" * 40)
        print(f"Loaded Video Shape: {video.shape}")
        print(f"Loaded Label: {label}")
        print("If shape is [16, 3, 224, 224], your data pipeline is ready!")
    else:
        print("Waiting for dataset to finish downloading...")