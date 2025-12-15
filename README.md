# 🎵 Deshimmer

**Remove the "sparkle" from AI-generated music**

A practical audio processing tool designed to detect and suppress the characteristic high-frequency artifacts ("shimmer" or "birdies") commonly found in Suno AI and other AI-generated music.

## 🎯 What It Does

AI music generation systems like Suno AI often produce narrow-band, flickering artifacts in the 5-7 kHz range that give the audio an unnatural, crystalline quality. **Deshimmer** targets these artifacts specifically:

- 🎚️ **Smart Detection**: Identifies narrow, flickering outliers above a local median baseline in target frequency bands
- 🔇 **Selective Suppression**: Applies soft-knee attenuation only to detected artifacts
- 🎼 **Music-Aware**: Protects transients and broadband musical content from processing
- 🌊 **Optional De-crystallization**: Adds subtle random-phase blending in noise-like frames to reduce the "digital shimmer"

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
python deshimmer.py input.wav output.wav
```

That's it! The default settings target the typical 5.1-7.2 kHz "shimmer zone" common in Suno AI output.

### Advanced Usage

```bash
# Target a specific frequency band
python deshimmer.py input.wav output.wav --start-hz 5500 --end-hz 7000

# Use center frequency and width in cents
python deshimmer.py input.wav output.wav --center-hz 6062 --width-cents 600

# More aggressive artifact removal
python deshimmer.py input.wav output.wav --thr-db 6.0 --slope 0.8

# Blend with original (50% wet/dry)
python deshimmer.py input.wav output.wav --mix 0.5

# Add noise resynthesis for extra de-crystallization
python deshimmer.py input.wav output.wav --noise-resynth 0.3

# Save the difference signal (what was removed)
python deshimmer.py input.wav output.wav --write-diff artifacts.wav

# Enable debug output
python deshimmer.py input.wav output.wav --debug
```

## 🔬 Analysis Tools

### Birdie Probe

Visualize and extract artifacts from a specific time region:

```bash
python birdie_probe.py input.wav --outdir probe_output --t0 23.0 --dur 4.0
```

This generates:
- **`artifact.wav`**: Isolated artifact audio
- **`roi_band_spectrogram.png`**: Frequency content in the target band
- **`roi_residual_map.png`**: Heatmap of artifacts above the local baseline

Perfect for analyzing problem areas before processing or validating the results.

## 🎛️ Parameters Explained

### Frequency Band
- `--start-hz` / `--end-hz`: Direct frequency range (default: 5100-7200 Hz)
- `--center-hz` / `--width-cents`: Alternative band specification using musical intervals
- `--edge-hz`: Smoothing at band edges to avoid artifacts (default: 200 Hz)

### Detection
- `--freq-med-bins`: Median filter width for local baseline (default: 9)
- `--thr-db`: Threshold above baseline to consider an outlier (default: 8.0 dB)
- `--flat-start` / `--flat-end`: Spectral flatness range for noise detection (0.25-0.70)

### Suppression
- `--slope`: Attenuation slope for detected artifacts (default: 0.6)
- `--density-lo` / `--density-hi`: Density thresholds to distinguish narrow artifacts from broadband content
- `--flux-thr-db` / `--flux-range-db`: Transient protection via energy flux detection

### Creative Controls
- `--noise-resynth`: Random-phase blend amount in noise-like frames (0.0-1.0, default: 0.0)
- `--mix`: Wet/dry blend (1.0 = full processing, 0.0 = bypass)

### Technical
- `--n-fft`: FFT size (default: 2048)
- `--hop`: Hop size in samples (default: 512)
- `--fade-ms`: Fade duration at start/end (default: 5.0 ms)
- `--no-pad`: Disable padding (may cause boundary clicks)

## 🧪 How It Works

1. **STFT Analysis**: Audio is analyzed via Short-Time Fourier Transform with overlapping windows
2. **Band Isolation**: Processing focuses on the target frequency band (typically 5.1-7.2 kHz)
3. **Local Baseline**: A frequency-wise median filter establishes what's "normal" at each time
4. **Outlier Detection**: Bins that spike significantly above the local baseline are flagged
5. **Context Awareness**:
   - **Spectral Flatness**: More processing on noise-like content, less on tonal/musical content
   - **Transient Protection**: Energy flux detection preserves drum hits and attacks
   - **Density Check**: Broadband musical events are protected from suppression
6. **Soft-Knee Attenuation**: Detected artifacts are smoothly attenuated, not brutally gated
7. **Optional Phase Randomization**: Subtle random-phase blend breaks up remaining crystalline patterns
8. **Reconstruction**: Overlap-add synthesis with careful normalization and fade-out

## 🎨 Use Cases

- **Post-process Suno AI tracks** to remove characteristic shimmer
- **Clean up codec artifacts** from lossy compression
- **Research tool** for studying AI music generation artifacts
- **A/B comparison** using the `--write-diff` option to hear exactly what was removed

## 📊 Example Results

Check the included `lab_*` directories for before/after spectrograms and metrics from test processing runs.

## 🤝 Contributing

This is an experimental tool! If you find parameter combinations that work well (or poorly) for specific types of content, please share your findings.

## 📝 License

MIT License - feel free to use, modify, and distribute.

## ⚠️ Notes

- **Best for high-quality sources**: Works best on 44.1 kHz or higher sample rates
- **Subtle by design**: Default settings are conservative; increase `--slope` or decrease `--thr-db` for more aggressive processing
- **Not a magic bullet**: Some artifacts may be too intertwined with the musical content to remove cleanly
- **Listen critically**: Always A/B test your results with the original

## 🎧 Happy De-shimmering!

Made with frustration (at AI music artifacts) and Python.
