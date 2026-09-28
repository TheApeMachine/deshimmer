# Tonal self-consistency

The default processor, API, CLI, previews, and WAV renderer all run the same
tonal repair stage after the existing spectral repairs and before wet/dry
blending and delivery mastering. `Params.tonal_repair=0.5` controls the amount;
`tonal_max_shift_cents=25` bounds each correction before applying that amount.
These are conservative product defaults, not inferred perceptual thresholds.
`mix=0` still bypasses all repair exactly.

**Set from audio** activates the default amount, including when loading an older
or previously disabled preset, and reports harmonic evidence in the selected
regions. **Optimize** evaluates six objectives and includes a final search over
tonal amount from zero to one. It retains the incoming candidate when proposals
score worse. A zero amount can win when the measurements favor preservation;
analysis and the coherence objective remain active. Output and diff downloads
use the selected parameters, and their sum reproduces the input within WAV
quantization precision. No new switch is needed in the normal workflow.

## Mechanism

`PartialTracker` estimates resolved sinusoidal frequency, complex amplitude, and
phase using a Hann window, fourfold zero padding, and quadratic log-magnitude
peak interpolation. A fitted finite-window Hann lobe rejects poorly resolved
peaks. Stereo detection combines channel powers rather than summing waveforms,
so opposite phases do not cancel. Lobe confidence and persistence also apply
separately to each channel. Tracks use one-to-one frequency assignment and end
when evidence disappears; missing spans are never predicted or filled.

`HarmonicField` proposes families using approximate harmonic relationships, then
fits `log2(frequency) = learned partial offset + shared motion` using alternating
medians. The offsets are learned independently; they are not rounded to integer
harmonics or a tuning grid. Correction requires a stable majority of simultaneous
partial trajectories. Stable peer relationships protect independent voices, and
competing stable cores mark shared partials as ambiguous. Only deviations beyond
both the configured consistency floor and the observed phase/frequency
uncertainty are eligible.

`TonalRepair` integrates the bounded frequency change into continuous phase and
uses cubic Hermite interpolation to preserve measured phase and frequency at
frame centers. It adds `corrected sinusoid - original modeled sinusoid` directly
to the input. Per-channel support boundaries use a half-window sine-squared
fade. Unsupported regions retain their original samples. This preserves the
unmodeled residual algebraically; it does not claim perfect stem separation or
an exact tonal/transient/noise decomposition.

The new optimizer objective compares output trajectories against the **input's**
families and expected motion. Refitting a different family in the output cannot
improve its target. Each partial's benefit is its absolute disagreement reduction,
normalized by the family's input mean disagreement plus the consistency floor,
weighted by retained partial power. Missing power incurs a proportional loss
penalty. The aggregate is expressed in percentage points and multiplied by the
configured coherence weight. This is an optimization utility, not a calibrated
audio-quality percentage. An input without supported families has no measured
correction opportunity and contributes zero; the report explicitly says why.

## Configuration and reporting

`TonalConfig` contains all analysis limits as named engineering choices:

| Setting | Default | Purpose |
|---|---:|---|
| `n_fft`, `hop`, `zero_padding` | 4096, 1024, 4 | Time/frequency resolution and peak interpolation |
| `min_hz`, `max_hz` | 40, 16000 Hz | Analysis band, further limited by resolved FFT support |
| `prominence_db`, `dynamic_range_db` | 12, 60 dB | Peak evidence above the local/background floor |
| `max_lobe_error` | 0.15 | Maximum relative Hann-lobe fit residual |
| `max_step_cents` | 100 | Maximum consecutive-frame assignment distance |
| `min_track_frames` | 4 | Minimum continuous observations, also per channel |
| `family_tolerance_cents` | 50 | Family proposal and input/output matching tolerance |
| `consistency_cents` | 2 | Minimum disagreement needed to justify correction |
| `stable_fraction`, `min_family_partials` | 0.75, 4 | Required stable majority and family support |
| `fit_iterations` | 8 | Bounded alternating-median fit iterations |

Reports include tracked partials, supported families, ambiguity counts,
median/p95 disagreement, and supported/unattached partial energy fractions.
Those fractions describe **tracked sinusoidal model energy**, not the proportion
of the entire waveform that is tonal. Unavailable measurements are `None`, not
fabricated zero-error results. Render metrics describe this stage before later
mixing/mastering; the optimizer separately measures the candidate output.

## Scope and verification

The implementation preserves the tested A=442 family, shared vibrato, stretched
harmonics, detuned chorus, and two independently vibrating voices. A known
wandering partial is corrected without shifting its agreeing peers. Tests also
exercise noise, impulses, missing fundamentals, repeated notes, channel gaps,
partial-energy loss, and the actual analyze/optimize/render/export callbacks.
Benchmarks run those regimes and the existing real-file processing/scoring paths.

Dense unresolved mixtures, short notes, and strongly inharmonic material can lack
usable family evidence and remain untouched by this stage. Stable mistuning is
indistinguishable from an intentional stable ratio in this model and is preserved.
This first implementation has no musical intonation normalization, instrument
identification, or prediction through masked passages. The fixtures establish
these implementation contracts, not perceptual superiority on unseen music.

Method references: Julius O. Smith's [Spectrum Analysis of Sinusoids](https://www.dsprelated.com/freebooks/SASP/Spectrum_Analysis_Sinusoids.html)
and [Spectral Interpolation](https://dsprelated.com/freebooks/sasp/Spectral_Interpolation.html)
describe the sinusoidal estimation basis; UPF's [Spectral Modeling Synthesis](https://www.upf.edu/web/mtg/sms-tools)
describes sinusoidal/residual synthesis. The family confidence rules and optimizer
utility here are project choices, not claims that these sources validate them.
