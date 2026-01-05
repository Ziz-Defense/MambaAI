
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from mamba_unfolding_net import MambaDOANet

# Mock Dataset
class MockCovarianceDataset(Dataset):
    def __init__(self, length=1000):
        self.length = length
        # MambaDOANet expects (Batch, Time, 16, 16*2) flatten input?
        # In my train script I did: 
        # R_seq = self.compute_scm(x_t) -> (T, 16, 16) complex
        # In MambaDOANet forward:
        # x = torch.cat([R_seq.real.flatten(2), R_seq.imag.flatten(2)], dim=2)
        
        # So output of dataset should be (T, 16, 16) complex
        self.T = 24 # From previous verification
        self.shape = (self.T, 16, 16)
        
    def __len__(self): return self.length
    def __getitem__(self, i):
        # Return random complex covariance
        return torch.randn(self.shape, dtype=torch.cfloat), torch.randn(2)

def benchmark():
    print("Benchmarking Mamba Training Speed...")
    
    # Model
    model = MambaDOANet(num_mics=16, d_model=128, grid_size=720).cuda()
    
    # Wrap in Regressor (as used in script)
    class MambaRegressor(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
            self.net.unfolding = nn.Identity()
            self.head = nn.Linear(128, 2)
        def forward(self, x):
            return self.head(self.net(x))
            
    model = MambaRegressor(model).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = nn.MSELoss()
    
    # Loader
    BATCH_SIZE = 64
    ds = MockCovarianceDataset(length=640) # 10 batches
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    
    # Warmup
    print("Warmup...")
    for x, y in loader:
        _ = model(x.cuda())
        break
        
    torch.cuda.synchronize()
    start_time = time.time()
    
    total_samples = 0
    print("Running...")
    for x, y in loader:
        optimizer.zero_grad()
        out = model(x.cuda())
        loss = criterion(out, y.cuda())
        loss.backward()
        optimizer.step()
        total_samples += BATCH_SIZE
        
    torch.cuda.synchronize()
    duration = time.time() - start_time
    
    throughput = total_samples / duration
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Time per 100k samples: {100000/throughput/60:.2f} minutes")

if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark()
    else:
        print("No GPU available for benchmark.")
