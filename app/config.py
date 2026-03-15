"""
Configuration for the Empathy Engine service.
"""

import os
from pathlib import Path
from typing import Optional

# Inject FFMPEG_PATH into PATH at import so pydub/subprocess find ffmpeg
# Set to the folder containing ffmpeg.exe, or the unpacked ffmpeg folder (we look for bin/)
_ffmpeg_path = os.getenv("FFMPEG_PATH")
if _ffmpeg_path:
    _p = Path(_ffmpeg_path).resolve()
    if _p.is_dir():
        if (_p / "ffmpeg.exe").exists() or (_p / "ffmpeg").exists():
            _bin = _p
        elif (_p / "bin" / "ffmpeg.exe").exists():
            _bin = _p / "bin"
        else:
            _bin = _p
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ.get("PATH", "")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "generated_audio"

# Emotion detection
EMOTION_MODEL_HF = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_CATEGORY = {
    "joy": "happy",
    "anger": "frustrated",
    "disgust": "frustrated",
    "fear": "frustrated",
    "sadness": "frustrated",
    "surprise": "happy",
    "neutral": "neutral",
}

# TTS
TTS_ENGINE = os.getenv("TTS_ENGINE", "gtts")  # gtts | elevenlabs
ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

# Audio
DEFAULT_SAMPLE_RATE = 22050
AUDIO_FORMAT = "mp3"
DEFAULT_VOLUME_DB = 0.0

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def ensure_output_dir() -> Path:
    """Create output directory if it does not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
