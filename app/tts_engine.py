"""
Text-to-speech engine with support for pitch, rate, and volume adjustment via pydub.
"""

import io
import tempfile
from pathlib import Path
from typing import Optional

from pydub import AudioSegment

from app.config import TTS_ENGINE, ensure_output_dir
from app.utils import get_logger, require_ffmpeg
from app.voice_mapper import SpeechParams

logger = get_logger(__name__)


def _synthesize_gtts(text: str, lang: str = "en") -> AudioSegment:
    """Generate speech using gTTS and return as pydub AudioSegment."""
    from gtts import gTTS

    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts.save(f.name)
        audio = AudioSegment.from_mp3(f.name)
    Path(f.name).unlink(missing_ok=True)
    return audio


def _synthesize_elevenlabs(text: str) -> AudioSegment:
    """Generate speech using ElevenLabs API and return as pydub AudioSegment."""
    from app.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID

    if not ELEVENLABS_API_KEY:
        raise ValueError("ELEVENLABS_API_KEY environment variable is not set")

    try:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.text_to_speech.convert(ELEVENLABS_VOICE_ID, text=text)
        data = b"".join(response)
        return AudioSegment.from_mp3(io.BytesIO(data))
    except Exception as e:
        logger.error("ElevenLabs TTS failed: %s", e)
        raise


def _apply_params(audio: AudioSegment, params: SpeechParams) -> AudioSegment:
    """Apply pitch, rate, and volume to an AudioSegment using pydub."""
    # Volume: apply gain in dB
    if params.volume_db != 0:
        audio = audio + params.volume_db

    # Pitch: change frame rate then set back to original (keeps duration, changes pitch)
    if params.pitch != 1.0:
        import math

        octaves = math.log2(params.pitch)
        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
        pitched = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": new_sample_rate},
        )
        pitched = pitched.set_frame_rate(audio.frame_rate)
        audio = pitched

    # Rate: speed up or slow down via frame_rate override
    if params.rate != 1.0:
        audio = audio._spawn(
            audio.raw_data,
            overrides={"frame_rate": int(audio.frame_rate * params.rate)},
        )

    return audio


def synthesize_speech(
    text: str,
    params: SpeechParams,
    lang: str = "en",
    engine: Optional[str] = None,
) -> AudioSegment:
    """
    Generate speech from text with the given speech parameters.
    Requires ffmpeg on PATH for MP3 handling.
    """
    require_ffmpeg()
    engine = engine or TTS_ENGINE
    if engine == "elevenlabs":
        raw = _synthesize_elevenlabs(text)
    else:
        raw = _synthesize_gtts(text, lang=lang)

    return _apply_params(raw, params)


def save_audio(audio: AudioSegment, file_path: Path, format: str = "mp3") -> Path:
    """
    Save an AudioSegment to disk.

    Args:
        audio: pydub AudioSegment to save.
        file_path: Destination path (file path or directory).
        format: Output format: 'mp3' or 'wav'.

    Returns:
        Resolved path to the saved file.
    """
    ensure_output_dir()
    path = Path(file_path)
    if path.suffix not in (".mp3", ".wav"):
        path = path.with_suffix(f".{format}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".wav":
        audio.export(str(path), format="wav")
    else:
        audio.export(str(path), format="mp3", bitrate="192k")

    logger.info("Saved audio to %s", path)
    return path
