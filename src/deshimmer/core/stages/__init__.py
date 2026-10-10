"""Individual modular DSP processing stages."""

from .base import DSPStage
from .expander import ExpanderConfig, ExpanderStage
from .denoise import SmartDenoiseConfig, SmartDenoiseStage
from .deresonator import DeResonatorConfig, DeResonatorStage
from .shimmer import ShimmerConfig, ShimmerStage
from .gain import GainApplicationConfig, GainApplicationStage
from .carver import CarverConfig, DynamicCarverStage
from .noise_resynth import NoiseResynthConfig, NoiseResynthStage
from .phase_blur import PhaseBlurConfig, PhaseBlurStage
from .swish import SwishRepairConfig, SwishRepairStage
from .decorrelation import HFDecorrelationConfig, HFDecorrelationStage
from .tonal_balance import TonalBalanceConfig, TonalBalanceStage
from .transient_punch import TransientPunchConfig, TransientPunchStage
from .low_end import LowEndConfig, LowEndStage
from .stereo_translation import StereoTranslationConfig, StereoTranslationStage
from .harmonic_enhance import HarmonicEnhanceConfig, HarmonicEnhanceStage

__all__ = [
    "DSPStage",
    "ExpanderConfig",
    "ExpanderStage",
    "SmartDenoiseConfig",
    "SmartDenoiseStage",
    "DeResonatorConfig",
    "DeResonatorStage",
    "ShimmerConfig",
    "ShimmerStage",
    "GainApplicationConfig",
    "GainApplicationStage",
    "CarverConfig",
    "DynamicCarverStage",
    "NoiseResynthConfig",
    "NoiseResynthStage",
    "PhaseBlurConfig",
    "PhaseBlurStage",
    "SwishRepairConfig",
    "SwishRepairStage",
    "HFDecorrelationConfig",
    "HFDecorrelationStage",
    "TonalBalanceConfig",
    "TonalBalanceStage",
    "TransientPunchConfig",
    "TransientPunchStage",
    "LowEndConfig",
    "LowEndStage",
    "StereoTranslationConfig",
    "StereoTranslationStage",
    "HarmonicEnhanceConfig",
    "HarmonicEnhanceStage",
]
