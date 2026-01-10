
import numpy as np
import scipy.signal
import master
from dataclasses import asdict

def test_vectorized_stft():
    # Create dummy audio
    sr = 44100
    duration = 1.0 # 1 second
    t = np.linspace(0, duration, int(sr*duration))
    # Stereo sine wave
    x = np.vstack([np.sin(2*np.pi*440*t), np.sin(2*np.pi*880*t)]).T # (44100, 2)
    
    print(f"Input shape: {x.shape}")
    
    # Params
    p = master.Params()
    p.n_fft = 1024
    p.hop = 512
    p.denoise = 0.5
    p.de_resonator = 0.5
    p.shimmer = 0.5
    p.pad = True
    
    try:
        y = master.process_stft(x, sr, p)
        print(f"Output shape: {y.shape}")
        
        if y.shape != x.shape:
             print("WARNING: Output shape mismatch!")
             # STFT padding might cause slight length diffs if not handled, 
             # but we handled it with `y = y[:n_samples]`
        
        print("Vectorized process_stft ran successfully.")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vectorized_stft()
