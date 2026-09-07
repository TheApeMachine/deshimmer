"""Time actual repair/scoring paths using a supplied audio file, with no quality proxy."""

import argparse
import statistics
import time
from dataclasses import replace

import numpy as np
import soundfile as sf
from scipy.signal import correlate

import auto_tune
import master
from deshimmer_api import process_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--seconds", type=float, default=3.0, help="Audio duration for each measurement")
    parser.add_argument("--repeat", type=int, default=3, help="Timed repetitions after one warmup")
    parser.add_argument("--compare-correlation", action="store_true",
                        help="Also time direct correlation as a computational reference")
    args = parser.parse_args()

    if args.seconds <= 0 or args.repeat < 1:
        parser.error("seconds and repeat must be positive")

    with sf.SoundFile(args.input) as source:
        sample_rate = source.samplerate
        audio = source.read(int(args.seconds * sample_rate), dtype="float32", always_2d=True)

    if len(audio) < 2048 or audio.shape[1] < 2:
        parser.error("benchmark requires stereo audio with at least 2048 samples")

    params = master.Params(swish_repair=0.3, denoise=0.2, deres=0.2)
    weights = auto_tune.weights_from_aggressiveness(0.5)
    processed, _ = process_audio(audio, sample_rate, params=params)
    band = (params.start_hz, params.end_hz)
    swish_band = (params.swish_start_hz, params.swish_end_hz)
    paths = {
        "align_lr_delay": lambda: master.align_lr_delay(audio, sample_rate),
        "process_audio": lambda: process_audio(audio, sample_rate, params=params),
        "bypass": lambda: process_audio(audio, sample_rate, params=replace(params, mix=0.0)),
        "residual_audit": lambda: process_audio(audio, sample_rate, params=params, include_residuals=True),
        "score_objectives": lambda: auto_tune.score_objectives(
            audio, processed, sample_rate, band=band, swish_band=swish_band, weights=weights,
        ),
    }

    if args.compare_correlation:
        # Reference only: the previous direct operation, on identical real samples.
        paths["direct_correlation_reference"] = lambda: np.correlate(audio[:, 0], audio[:, 1], mode="same")
        direct = paths["direct_correlation_reference"]()
        fft = correlate(audio[:, 0], audio[:, 1], mode="same", method="fft")

        if np.argmax(direct) != np.argmax(fft):
            raise AssertionError("direct and FFT correlations select different channel delays")

    print(f"input={args.input} rate={sample_rate} samples={len(audio)} channels={audio.shape[1]} repeat={args.repeat}")

    for name, operation in paths.items():
        operation()
        elapsed = []

        for _ in range(args.repeat):
            started = time.perf_counter()
            operation()
            elapsed.append(time.perf_counter() - started)

        median = statistics.median(elapsed)
        print(f"{name}: median_ms={median * 1000:.3f} realtime_factor={median / (len(audio) / sample_rate):.6f}")


if __name__ == "__main__":
    main()
