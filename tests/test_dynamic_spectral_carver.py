import warnings

import numpy as np

import master


def test_dynamic_spectral_carver_broadcasts_gain_over_channels():
    freqs = np.linspace(0.0, 24_000.0, 257, dtype=np.float32)
    Z = np.ones((freqs.size, 4, 2), dtype=np.complex64)
    Z[120:180, :, :] *= 40.0

    y = master.dynamic_spectral_carver(Z, freqs)

    assert y.shape == Z.shape
    assert np.all(np.isfinite(y))


def test_psychoacoustic_mask_ignores_sub_20hz_ath_overflow():
    freqs = np.linspace(0.0, 24_000.0, 257, dtype=np.float32)
    psd = np.full((freqs.size, 3), 1e-6, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mask = master._calculate_psychoacoustic_mask(psd, freqs)

    assert mask.shape == psd.shape
    assert np.all(np.isfinite(mask))


if __name__ == "__main__":
    test_dynamic_spectral_carver_broadcasts_gain_over_channels()
    test_psychoacoustic_mask_ignores_sub_20hz_ath_overflow()
