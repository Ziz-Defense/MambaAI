import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

# --- 1. MAMBA BLOCK (The 2025 "Denoising Engine") ---
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, bias=True, 
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv-1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32), 
            "n -> d n", 
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, u):
        # u: (Batch, SeqLen, Dim)
        batch, seq_len, d_model = u.shape
        
        x_and_z = self.in_proj(u)
        x_and_z = rearrange(x_and_z, "b l d -> b d l")
        x, z = x_and_z.chunk(2, dim=1)

        # Conv1d processing
        x_conv = self.conv1d(x)[:, :, :seq_len]
        x_conv = F.silu(x_conv)

        # SSM Parameters
        x_ssm_in = rearrange(x_conv, "b d l -> b l d")
        x_dbl = self.x_proj(x_ssm_in)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log.float())
        h = torch.zeros(batch, self.d_inner, self.d_state, device=u.device)
        
        y_ssm = []
        # Sequential scan (can be parallelized with associative scan in advanced versions)
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1) # (B, D, 1)
            dA = torch.exp(A * dt_t)         # (B, D, N)
            dB = dt_t * B[:, t, :].unsqueeze(1) # (B, D, N)
            
            x_t = x_ssm_in[:, t, :].unsqueeze(-1) # (B, D, 1)
            
            h = dA * h + dB * x_t # State Update: h_t = A*h_{t-1} + B*x_t
            
            # Output projection: y_t = C*h_t
            y_t = torch.sum(h * C[:, t, :].unsqueeze(1), dim=-1)
            y_ssm.append(y_t)

        y_ssm = torch.stack(y_ssm, dim=1)
        y_ssm = y_ssm + x_ssm_in * self.D.unsqueeze(0)
        
        # Gate and Output
        out = y_ssm * F.silu(rearrange(z, "b d l -> b l d"))
        return self.out_proj(out)

# --- 2. DEEP UNFOLDING / ADMM HEAD (The "Physics Engine") ---
class ADMMUnfoldingLayer(nn.Module):
    def __init__(self, num_mics, num_angles=360):
        super().__init__()
        # Learnable parameters for ADMM steps
        self.rho = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # The Observation Matrix (Dictionary) D
        # In a fully physics model, this is frozen. In Deep Unfolding, we LEARN it to correct calibration.
        # Initialize with random or theory, then let it evolve.
        self.D_real = nn.Parameter(torch.randn(num_mics, num_angles) * 0.1)
        self.D_imag = nn.Parameter(torch.randn(num_mics, num_angles) * 0.1)

    def soft_threshold(self, x, lmbda):
        return torch.sign(x) * F.relu(torch.abs(x) - lmbda)

    def forward(self, y, x_prev, z_prev, u_prev):
        # y: Input features (Measurements)
        # x: Sparse Angle Vector (What we want)
        # z: Auxiliary variable
        # u: Dual variable
        
        # Solving: min ||y - Dx|| + lambda|x|
        # Ideally implemented as complex operations. Approximating with 2-channel real for gradients.
        
        # 1. x-update (Least Squares + Lagrangian)
        # x = (D^T D + rho I)^-1 (D^T y + rho(z - u))
        # For efficiency in a layer, we can learn the inversion as a Linear layer 'W'
        pass # To be fleshed out in full network

class LearnedISTA(nn.Module):
    """
    Simplified Unfolding: Learned Iterative Shrinkage-Thresholding Algorithm.
    Often more stable than full ADMM for Neural Nets.
    x_{k+1} = soft(x_k + S (y - D x_k), theta)
    """
    def __init__(self, input_dim, grid_size=360, layers=10):
        super().__init__()
        self.layers = layers
        self.S = nn.ModuleList([nn.Linear(input_dim, grid_size) for _ in range(layers)])
        self.D = nn.ModuleList([nn.Linear(grid_size, input_dim) for _ in range(layers)]) # The learned physics matrix
        self.theta = nn.Parameter(torch.ones(layers) * 0.01) # Learnable thresholds

    def forward(self, y):
        # y: (Batch, InputDim) - Encoded Covariance
        batch = y.shape[0]
        x = torch.zeros(batch, self.S[0].out_features, device=y.device) # Initialize x=0
        
        for i in range(self.layers):
            # Gradient Descent Step: (y - Dx) is the residual error
            residual = y - self.D[i](x)
            
            # Gradient Step + Shrinkage
            update = self.S[i](residual)
            x = x + update
            
            # Soft Thresholding (Sparsity Enforcement)
            x = torch.sign(x) * F.relu(torch.abs(x) - self.theta[i])
            
        return x # (Batch, grid_size) -> Angular Spectrum

# --- 3. HYBRID MODEL ---
class MambaDOANet(nn.Module):
    def __init__(self, num_mics=16, d_model=128, grid_size=720): # 0.5 deg resolution grid
        super().__init__()
        
        # Input: Covariance Matrix (16x16 complex) -> Flattened 512
        self.input_proj = nn.Linear(512, d_model)
        
        # Mamba Backbone (Denoising & Feature Extraction)
        # Processes the sequence of covariances over time/freq to get a robust snapshot
        self.mamba = MambaBlock(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Deep Unfolding Head (Physics Solver)
        # Maps robust features to Angular Grid
        self.unfolding = LearnedISTA(input_dim=d_model, grid_size=grid_size, layers=12)
        
    def forward(self, R_seq):
        # R_seq: (Batch, Time, 16, 16) Complex
        # Flatten spatial dims: (B, T, 512)
        B, T, _, _ = R_seq.shape
        x = torch.cat([R_seq.real.flatten(2), R_seq.imag.flatten(2)], dim=2)
        
        # Embed
        x = self.input_proj(x) # (B, T, D)
        
        # Mamba Process
        x = self.mamba(x)
        x = self.norm(x)
        
        # Pool temporal dimension (Average the denoised states)
        x_pool = x.mean(dim=1) # (B, D)
        
        # Unfold to DOA
        spectrum = self.unfolding(x_pool) # (B, GridSize)
        return spectrum
