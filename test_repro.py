
import numpy as np
from scipy import signal

def test_stft_istft():
    # Simulate the scenario
    sr = 44100
    x = np.random.randn(100000, 2).astype(np.float32) # (T, C)
    n_fft = 2048
    hop = 512
    
    print(f"Input x shape: {x.shape}")
    
    # STFT
    win = signal.get_window("hann", n_fft, fftbins=True).astype(np.float32)
    noverlap = n_fft - hop
    
    f, tt, Z = signal.stft(
        x,
        fs=sr,
        window=win,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        boundary="zeros",
        padded=True,
        axis=0,
    )
    
    print(f"Z shape: {Z.shape}") # Expected: (F, T_seg, C)
    
    # ISTFT
    _, y = signal.istft(
        Z,
        fs=sr,
        window=win,
        nperseg=n_fft,
        noverlap=noverlap,
        nfft=n_fft,
        input_onesided=True,
        boundary=True,
        time_axis=1,
        freq_axis=0,
    )
    
    print(f"Output y shape: {y.shape}")

if __name__ == "__main__":
    test_stft_istft()

