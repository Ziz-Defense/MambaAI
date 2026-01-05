
import torch
import numpy as np

def compute_scm(x_chunk, n_fft=1024, hop=512):
    # x_chunk: (16, N_samples)
    x_tensor = torch.tensor(x_chunk, dtype=torch.float32)
    win = torch.hann_window(n_fft)
    
    # 1. STFT
    # (16, F, T)
    X = torch.stft(x_tensor, n_fft, hop, window=win, return_complex=True)
    
    # 2. Covariance
    X = X.permute(2, 0, 1) # (T, M, F)
    T, M, F = X.shape
    
    # (T, M, F) -> (T, F, M)
    X_tfm = X.permute(0, 2, 1) # (T, F, M)
    
    # Batch matmul over T*F
    X_flat = X_tfm.reshape(T*F, M, 1)
    
    # R = X @ X^H
    R_flat = torch.matmul(X_flat, X_flat.conj().transpose(1, 2)) # (TF, M, M)
    R_flat = R_flat.reshape(T, F, M, M)
    
    # Average over Frequency
    R_seq = R_flat.mean(dim=1) # (T, M, M)
    
    return R_seq

def test_scm():
    print("Testing SCM Computation...")
    # Mock audio: 16 channels, 12000 samples (0.25s @ 48k)
    # Correlated signal + noise
    t = np.linspace(0, 0.25, 12000)
    sig = np.sin(2*np.pi*440*t) # 440Hz tone
    
    channels = []
    for i in range(16):
        # Shift phase slightly
        channels.append(np.sin(2*np.pi*440*t + i*0.1) + 0.01*np.random.randn(len(t)))
    
    audio = np.stack(channels) # (16, 12000)
    
    R = compute_scm(audio)
    print(f"Output Shape: {R.shape}")
    print(f"Dtype: {R.dtype}")
    
    # Check properties
    # Hermitian?
    err = (R - R.conj().transpose(1, 2)).abs().max()
    print(f"Hermitian Error: {err:.6f}")
    
    # Expected size
    # 12000 samples, hop 512 + padding -> ~24 frames
    expected_frames = 12000 // 512 + 1
    # STFT often produce N//H + 1
    print(f"Time Steps: {R.shape[0]}")
    
    assert R.shape[1:] == (16, 16)
    assert err < 1e-4
    print("✅ Logic Verified")

if __name__ == "__main__":
    test_scm()
