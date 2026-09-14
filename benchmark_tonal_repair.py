"""Real partial analysis/resynthesis on known preservation and detuning regimes."""

import argparse
import statistics
import time

import numpy as np

from tonal_fixtures import TonalFixture
from tonal_repair import TonalRepair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.seconds <= 0 or args.repeat < 1:
        parser.error("seconds and repeat must be positive")

    fixture = TonalFixture(duration=args.seconds)
    processor = TonalRepair()
    print(f"rate={fixture.sample_rate} seconds={args.seconds} channels=2 repeat={args.repeat}")

    for mode in ("coherent", "vibrato", "stretched", "chorus", "polyphonic", "noise", "transient", "wandering"):
        audio = fixture.render(mode, stereo=True)
        processor.process(audio, fixture.sample_rate)
        elapsed = []

        for _ in range(args.repeat):
            started = time.perf_counter()
            output, report = processor.process(audio, fixture.sample_rate)
            elapsed.append(time.perf_counter() - started)

        before = report["before"]["p95_disagreement_cents"]
        after = report["after"]["p95_disagreement_cents"]
        before_text = f"{before:.3f}" if before is not None else "n/a"
        after_text = f"{after:.3f}" if after is not None else "n/a"
        print(f"{mode}: changed={report['changed_partials']} p95_cents={before_text}->{after_text} "
              f"identical={np.array_equal(output, audio)} median_ms={statistics.median(elapsed)*1000:.3f}")


if __name__ == "__main__":
    main()
