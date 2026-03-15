"""
Map detected emotion to speech parameters (pitch, rate, volume).
"""

from dataclasses import dataclass
from typing import Literal

from app.emotion_detector import EmotionCategory, EmotionResult

# Multipliers: 1.0 = default
# pitch: ratio (e.g. 1.2 = +20%)
# rate: ratio (e.g. 1.1 = 10% faster)
# volume_db: delta in dB (e.g. +2.0 = louder)
EMOTION_PARAMS = {
    "happy": {"pitch": 1.2, "rate": 1.1, "volume_db": 2.0},
    "frustrated": {"pitch": 0.9, "rate": 0.85, "volume_db": -2.0},
    "neutral": {"pitch": 1.0, "rate": 1.0, "volume_db": 0.0},
}


@dataclass
class SpeechParams:
    """Speech parameters derived from emotion (and optional intensity)."""

    pitch: float  # multiplier, 1.0 = default
    rate: float   # multiplier, 1.0 = default
    volume_db: float  # delta in dB

    def __str__(self) -> str:
        return f"pitch={self.pitch:.2f}, rate={self.rate:.2f}, volume_db={self.volume_db:.2f}"


def _scale_intensity(base: float, default: float, intensity: float) -> float:
    """Scale a parameter toward default based on intensity in [0, 1]."""
    if intensity >= 1.0:
        return base
    # Lerp from default to base as intensity goes from 0 to 1
    return default + (base - default) * intensity


def get_speech_params(
    emotion_result: EmotionResult,
    intensity: float = 1.0,
) -> SpeechParams:
    """
    Map emotion (and optional intensity) to speech parameters.

    Intensity in [0, 1] scales the effect: 0 = neutral, 1 = full emotion mapping.

    Args:
        emotion_result: Result from detect_emotion().
        intensity: Optional scaling of emotion effect (default 1.0 = full).

    Returns:
        SpeechParams with pitch, rate, and volume_db.
    """
    emotion: EmotionCategory = emotion_result.emotion
    base = EMOTION_PARAMS.get(emotion, EMOTION_PARAMS["neutral"])
    confidence = max(0.0, min(1.0, emotion_result.confidence))
    effective_intensity = intensity * confidence

    pitch = _scale_intensity(base["pitch"], 1.0, effective_intensity)
    rate = _scale_intensity(base["rate"], 1.0, effective_intensity)
    volume_db = _scale_intensity(base["volume_db"], 0.0, effective_intensity)

    return SpeechParams(pitch=pitch, rate=rate, volume_db=volume_db)
