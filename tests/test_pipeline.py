"""
Basic tests for the Empathy Engine pipeline.
"""

import pytest
from pathlib import Path

from app.emotion_detector import detect_emotion, EmotionResult
from app.voice_mapper import get_speech_params, SpeechParams, EMOTION_PARAMS


def test_detect_emotion_returns_result():
    """detect_emotion returns an EmotionResult with expected fields."""
    result = detect_emotion("This is great news!", use_fallback_only=True)
    assert isinstance(result, EmotionResult)
    assert result.emotion in ("happy", "frustrated", "neutral")
    assert 0 <= result.confidence <= 1.0


def test_detect_emotion_empty_text():
    """Empty text yields neutral with zero confidence."""
    result = detect_emotion("")
    assert result.emotion == "neutral"
    assert result.confidence == 0.0

    result = detect_emotion("   ")
    assert result.emotion == "neutral"


def test_get_speech_params_mapping():
    """Speech params match the configured emotion mapping."""
    for emotion in ("happy", "frustrated", "neutral"):
        res = EmotionResult(emotion=emotion, raw_label=emotion, confidence=1.0)
        params = get_speech_params(res, intensity=1.0)
        expected = EMOTION_PARAMS[emotion]
        assert params.pitch == expected["pitch"]
        assert params.rate == expected["rate"]
        assert params.volume_db == expected["volume_db"]


def test_get_speech_params_intensity_scaling():
    """Lower intensity moves params toward neutral."""
    res = EmotionResult(emotion="happy", raw_label="joy", confidence=1.0)
    full = get_speech_params(res, intensity=1.0)
    half = get_speech_params(res, intensity=0.5)
    zero = get_speech_params(res, intensity=0.0)
    assert half.pitch < full.pitch and half.pitch > 1.0
    assert zero.pitch == 1.0 and zero.rate == 1.0 and zero.volume_db == 0.0


def _ffmpeg_available() -> bool:
    """Check if ffmpeg/ffprobe is available (required by pydub for MP3)."""
    import shutil
    return shutil.which("ffmpeg") is not None or shutil.which("avconv") is not None


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg/avconv required for MP3 in pydub")
def test_run_pipeline_integration(tmp_path):
    """Full pipeline produces an audio file and returns emotion."""
    from app.main import run_pipeline

    out = tmp_path / "test_out.mp3"
    emotion, confidence, path = run_pipeline("Hello, world!", output_path=out, lang="en")
    assert emotion in ("happy", "frustrated", "neutral")
    assert 0.0 <= confidence <= 1.0
    assert path == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_run_pipeline_rejects_empty_text():
    """run_pipeline raises ValueError for empty text."""
    from app.main import run_pipeline

    with pytest.raises(ValueError, match="empty"):
        run_pipeline("")


def test_fastapi_app_structure():
    """FastAPI app has required routes and returns expected fields."""
    from app.main import app
    from fastapi.testclient import TestClient

    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/generate-voice" in routes
    assert "/" in routes

    client = TestClient(app)
    resp = client.post("/generate-voice", json={"text": "Hello!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "emotion" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "audio_file" in data
