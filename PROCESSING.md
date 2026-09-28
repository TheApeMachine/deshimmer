# Repair, residuals, and optimizer verification

The normal **Set from audio → Optimize → Render full & generate downloads**
workflow includes [tonal self-consistency repair](TONAL_REPAIR.md) by default.
Analysis activates its initial amount; optimization evaluates its coherence
objective in every stage and searches its amount in a final tonal stage. No
additional checkbox, preset, or alternate renderer is required.
New repair stages must follow this workflow contract: participate in the normal
defaults, analysis, optimization, preview, and export without another enabling step.

`master.process_stft` blends the original input with the complete wet repair,
including optional HF resynthesis. `mix=0` returns the original samples and a
zero repair residual without running the DSP. Delivery mastering is separate:
disable it as the UI's Bypass preset does for an end-to-end null.

The API and CLI apply `delta_listen` after delivery mastering, so it returns
input minus the final processed output. Measurements describe the processed
output, even when listening to its residual.

In the UI, **Listen to repair-only diff** selects a separate repair render with
L/R alignment, spectral balancing, spectral carving, and delivery mastering
disabled. The normal output retains all configured processing. The selected diff
is used for preview and WAV downloads. **Listen to removed audio** swaps the
output and diff players/downloads, as before.

API callers can request both residuals:

```python
output, info = process_audio(audio, sample_rate, params=params, include_residuals=True)
repair_diff = info["residuals"]["artifact_repair_diff"]
total_diff = info["residuals"]["total_enhancement_diff"]
```

The repair-only comparison costs an additional repair pass when enhancement is
enabled. Its difference from the total residual includes interactions between
stages, so it is not an isolated recording of the engineering stages. Residual
arrays are omitted from `info` by default. `Params(enhance=False)` or the
`master.py --repair-only` option disables those three engineering stages in the
main output itself; delivery mastering remains independently configurable.

Refinement uses six weighted objectives: artifact reduction, music damage,
diff leakage, stereo damage, loudness/tilt damage, and tonal coherence. Every `Weights` term
contributes exactly once. Their sum is the same utility used for Pareto selection,
stage summaries, and final evaluation. Aggressiveness changes these weights and
retains preservation penalties at its upper endpoint. Musical-noise damage
measures added variation against the input in the same input-defined quiet frames.

Swish is measured over `swish_start_hz`–`swish_end_hz`; shimmer retains its own
band. Direct scoring calls must supply `swish_band` explicitly. Invalid or
undersized scoring inputs raise errors. Mathematical trial failures are logged
and pruned, never returned as apparently valid objective values. Each stage
includes the incoming settings as a candidate and retains them when proposals
score worse. Refining from bypass activates the wet mix that was evaluated.

Run regression tests with the test environment:

```sh
uv run pytest
# or: .venv/bin/pytest tests/
```

All 27 regression tests establish implementation behavior, channel alignment contracts, and numerical stability; they do not establish perceptual quality on an unseen track.

