
import cProfile
import pstats
import io
import numpy as np
import master

def profile_vectorized_stft():
    # Create dummy audio (10 seconds stereo)
    sr = 44100
    duration = 10.0
    t = np.linspace(0, duration, int(sr*duration))
    x = np.vstack([np.sin(2*np.pi*440*t), np.sin(2*np.pi*880*t)]).T
    
    # Params
    p = master.Params()
    p.n_fft = 2048
    p.hop = 512
    p.denoise = 0.5
    p.de_resonator = 0.5
    p.shimmer = 0.5
    p.pad = True
    
    # Warmup for Numba JIT
    try:
        print("Warming up JIT...")
        _ = master.process_stft(x, sr, p)
        print("Warmup complete.")
    except Exception as e:
        print(f"WARMUP FAILED: {e}")
        return

    pr = cProfile.Profile()
    pr.enable()
    
    try:
        y = master.process_stft(x, sr, p)
    except Exception as e:
        print(f"FAILED: {e}")
        return

    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20) # Top 20 lines
    print(s.getvalue())

if __name__ == "__main__":
    profile_vectorized_stft()
