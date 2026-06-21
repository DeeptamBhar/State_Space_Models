import os
import gzip
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba
import math
from tqdm import tqdm

# Downloading only chromosome 22 (~50 million bases)
CHR22_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
FILE_NAME = "chr22.fa.gz"

def download_and_tokenize():
    if not os.path.exists(FILE_NAME):
        print("Downloading Chromosome 22 from UCSC...")
        urllib.request.urlretrieve(CHR22_URL, FILE_NAME)
    
    print("Reading and tokenizing DNA...")
    vocab = {'N': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, 'a': 1, 'c': 2, 'g': 3, 't': 4}
    seq_tensor = []
    
    with gzip.open(FILE_NAME, 'rt') as f:
        f.readline() # Skip the header line 
        for line in f:
            for char in line.strip():
                if char in vocab:
                    seq_tensor.append(vocab[char])
                    
    # returning as a single flat tensor
    return torch.tensor(seq_tensor, dtype=torch.long)

# The mamba model using mamba_ssm
class GenomicMamba(nn.Module):
    def __init__(self, vocab_size=5, d_model=128, d_state=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        return self.head(x)

# Training and evaluation
def train_and_evaluate(data, d_model, seq_len, steps=200, batch_size=8):
    device = "cuda"
    model = GenomicMamba(d_model=d_model).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0
    
    for step in tqdm(range(steps)):
        # Random sequence sampling
        start_idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
        
        # Create X (input) and Y (target shifted by 1 to the left)
        x = torch.stack([data[i : i + seq_len] for i in start_idx]).to(device)
        y = torch.stack([data[i + 1 : i + seq_len + 1] for i in start_idx]).to(device)
        
        optimizer.zero_grad()
        logits = model(x)
     
        loss = criterion(logits.view(-1, 5), y.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / steps
    perplexity = math.exp(avg_loss)
    
    # Calculate parameter count
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return params, perplexity

if __name__ == "__main__":
    dna_data = download_and_tokenize()
    print(f"Total bases loaded: {len(dna_data):,}")
    
    # Experiment A: Params vs. Perplexity (Fixed SeqLen = 1024)
    print("\nRunning Exp A: Params vs Perplexity")
    d_models = [64, 128, 256] # Controls parameter count
    for d in d_models:
        params, pp = train_and_evaluate(dna_data, d_model=d, seq_len=1024)
        print(f"d_model: {d:3d} | Params: {params:9,d} | Perplexity: {pp:.4f}")

    # Experiment B: SeqLen vs. Perplexity (Fixed d_model = 128)
    print("\nRunning Exp B: SeqLen vs Perplexity")
    seq_lengths = [256, 1024, 2048, 4096] # Controls sequence length
    for sl in seq_lengths:
        try:
            _, pp = train_and_evaluate(dna_data, d_model=128, seq_len=sl, batch_size=4)
            print(f"SeqLen: {sl:4d} | Perplexity: {pp:.4f}")
        except RuntimeError as e: # Error handler for OOM on GPU
            if "Out of memory" in str(e):
                print(f"SeqLen: {sl:4d} | OOM Error (GPU limit reached)")
                torch.cuda.empty_cache()
            else:
                raise e