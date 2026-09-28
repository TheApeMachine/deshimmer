# Command-Line Interface (CLI) Manual

The Deshimmer command-line interface, located in [src/deshimmer/cli.py](file:///Users/theapemachine/go/src/github.com/theapemachine/deshimmer/src/deshimmer/cli.py), provides automated batch processing, headless cluster rendering, and integration into automated audio mastering and restoration pipelines.

---

## 1. Invocation Methods

You can invoke the CLI through several equivalent methods:

```bash
# Using Makefile
make cli ARGS="input.wav output.wav --thr-db 7.0"

# Using the installed console script
deshimmer-cli input.wav output.wav [options]

# Using python module directly
python -m deshimmer.cli input.wav output.wav [options]
```

---

## 2. Complete Arguments Reference

### Positional Arguments
- `input`: Path to input audio file (WAV, FLAC, MP3, OGG, AIFF).
- `output`: Path to output processed audio file.

### Core Shimmer Suppression
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--start-hz` | float | `5100.0` | Lower cutoff frequency for shimmer detection band. |
| `--end-hz` | float | `7200.0` | Upper cutoff frequency for shimmer detection band. |
| `--center-hz` | float | `None` | Alternative: set band center in Hz (used with `--width-cents`). |
| `--width-cents` | float | `None` | Bandwidth in musical cents around center frequency. |
| `--thr-db` | float | `8.0` | Flux threshold in dB above background baseline for attenuation. |
| `--slope` | float | `0.6` | Attenuation compression slope (higher = more aggressive). |
| `--density-lo` | float | `0.02` | Minimum energy density required to authorize attenuation. |
| `--density-hi` | float | `0.15` | Maximum energy density ceiling (prevents attenuation in dense choruses). |
| `--noise-resynth` | float | `0.0` | Random-phase noise injection level in noise-like frames (0.0 to 1.0). |
| `--mix` | float | `1.0` | Wet/dry blend (1.0 = fully processed, 0.0 = bypass null). |

### Tonal Self-Consistency
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--tonal-repair` | float | `0.5` | Amount of sinusoidal self-consistency correction (0.0 to 1.0). |
| `--tonal-max-shift-cents` | float | `25.0` | Maximum frequency deviation allowed per partial track in cents. |

### Minimum-Statistics Denoising
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--denoise` | float | `0.0` | Denoise depth factor (0.0 = disabled, 1.0 = full). |
| `--dn-start-hz` | float | `3000.0` | Lower frequency boundary for denoising. |
| `--dn-end-hz` | float | `16000.0` | Upper frequency boundary for denoising. |
| `--dn-floor-db` | float | `-24.0` | Noise floor safety limit in dB (prevents musical noise). |
| `--dn-minwin-ms` | float | `1000.0` | Time tracking window for minimum-statistics floor estimation. |
| `--dn-freq-smooth-bins` | int | `5` | Frequency smoothing radius in bins to prevent isolated chirps. |

### Smart De-Resonator (Dynamic Notch)
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--deres` | float | `0.0` | De-resonator depth (0.0 = disabled, 1.0 = maximum). |
| `--deq-start-hz` | float | `180.0` | Lower frequency limit for peak detection. |
| `--deq-end-hz` | float | `12000.0` | Upper frequency limit for peak detection. |
| `--deq-thr-db` | float | `4.0` | Peak prominence in dB above median background required for notch. |
| `--deq-max-att-db` | float | `12.0` | Maximum allowable attenuation per resonance peak in dB. |
| `--deq-persist-ms` | float | `1000.0` | Minimum duration in ms a peak must persist before attenuation. |
| `--deq-time-floor` | flag | `False` | Enable time-stabilized baseline floor tracking (protects vibrato). |

### Delivery Mastering
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--master` | flag | `False` | Enable the mastering normalization and true-peak limiter stage. |
| `--target-lufs` | float | `-14.0` | Integrated loudness target in LUFS (ITU-R BS.1770 / EBU R128). |
| `--ceiling-dbtp` | float | `-1.0` | True-peak limiter ceiling in dBTP. |
| `--tp-os` | int | `4` | Polyphase oversampling multiplier (4x recommended). |
| `--hp-hz` | float | `20.0` | High-pass DC/subsonic filter cutoff frequency in Hz. |

### Forensics & Diagnostics
| Flag | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--delta-listen` | flag | `False` | Render output as difference signal: $x_{\text{input}} - y_{\text{processed}}$. |
| `--repair-only` | flag | `False` | Disable L/R alignment, spectral balancing, and spectral carving. |
| `--write-diff` | str | `None` | Path to export a synchronized difference WAV alongside output. |
| `--debug` | flag | `False` | Enable diagnostic spectral data collection. |
| `--debug-dir` | str | `None` | Directory to save debug spectrogram PNGs and summary JSON. |
| `--subtype` | str | `PCM_24` | Soundfile export subtype (`PCM_16`, `PCM_24`, `FLOAT`). |

---

## 3. Practical CLI Recipes

### Recipe 1: Standard AI Shimmer Suppression
A conservative, safe pass targeting typical 5–7 kHz digital sparkle:
```bash
deshimmer-cli input.wav output.wav \
  --start-hz 5100 --end-hz 7200 \
  --thr-db 8.0 --slope 0.6 \
  --write-diff diff.wav
```

### Recipe 2: Suno / Udio 3.5 kHz Whine Removal
Targets the characteristic persistent whistle found in AI generated audio:
```bash
deshimmer-cli generated.mp3 cleaned.wav \
  --start-hz 3000 --end-hz 4200 \
  --deres 0.8 --deq-thr-db 4.0 --deq-max-att-db 12.0 \
  --deq-freq-med-bins 61 --deq-persist-ms 1000 \
  --thr-db 6.0 --slope 0.9 \
  --tonal-repair 0.6
```

### Recipe 3: Full Restoration Stack with Streaming Delivery Mastering
Executes full shimmer repair, light denoising, tonal repair, and delivers a mastered -14.0 LUFS file with -1.0 dBTP ceiling:
```bash
deshimmer-cli rough_mix.wav master_delivery.wav \
  --start-hz 5000 --end-hz 7500 \
  --thr-db 7.5 --slope 0.7 \
  --denoise 0.35 --dn-floor-db -22.0 \
  --tonal-repair 0.5 \
  --master --target-lufs -14.0 --ceiling-dbtp -1.0 \
  --subtype PCM_24
```

### Recipe 4: Batch Processing an Entire Directory
Bash one-liner to process all WAV files in a directory while creating synchronized difference files:
```bash
mkdir -p processed_masters
for f in raw_tracks/*.wav; do
  base=$(basename "$f" .wav)
  echo "Processing $base..."
  deshimmer-cli "$f" "processed_masters/${base}_repaired.wav" \
    --start-hz 5100 --end-hz 7200 \
    --deres 0.5 --denoise 0.25 \
    --write-diff "processed_masters/${base}_diff.wav" \
    --master --target-lufs -14.0
done
```
