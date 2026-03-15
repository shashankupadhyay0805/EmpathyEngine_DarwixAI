"""
Utility functions for the Empathy Engine.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from app.config import LOG_FORMAT, LOG_LEVEL, OUTPUT_DIR, ensure_output_dir


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level or LOG_LEVEL)
    return logger


def generate_audio_filename(extension: str = "mp3") -> Path:
    """Generate a unique filename for the output audio."""
    ensure_output_dir()
    name = f"{uuid.uuid4().hex[:12]}.{extension}"
    return OUTPUT_DIR / name


def _search_common_ffmpeg_locations() -> Optional[Path]:
    """Search common Windows/Unix locations for an ffmpeg bin directory."""
    candidates = []
    if os.name == "nt":
        user = os.environ.get("USERPROFILE", "")
        if user:
            downloads = Path(user) / "Downloads"
            if downloads.is_dir():
                # e.g. .../Downloads/ffmpeg-8.0.1-essentials_build/bin or .../ffmpeg-8.0.1-essentials_build.zip/.../bin
                for parent in downloads.iterdir():
                    if not parent.is_dir():
                        continue
                    name_lower = parent.name.lower()
                    if "ffmpeg" not in name_lower:
                        continue
                    bin_dir = parent / "bin"
                    if (bin_dir / "ffmpeg.exe").exists():
                        candidates.append(bin_dir)
                    elif (parent / "ffmpeg.exe").exists():
                        candidates.append(parent)
                    # Nested: e.g. ffmpeg-8.0.1-essentials_build.zip\ffmpeg-8.0.1-essentials_build\bin
                    for sub in parent.iterdir() if parent.is_dir() else []:
                        if sub.is_dir() and "ffmpeg" in sub.name.lower():
                            sub_bin = sub / "bin"
                            if (sub_bin / "ffmpeg.exe").exists():
                                candidates.append(sub_bin)
    if candidates:
        try:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        except OSError:
            return candidates[0]
    return None


def _find_ffmpeg_bin() -> Optional[Path]:
    """Return the directory containing ffmpeg.exe if found (PATH, FFMPEG_PATH, or common locations)."""
    import shutil
    exe = shutil.which("ffmpeg") or shutil.which("avconv")
    if exe:
        return Path(exe).resolve().parent
    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path:
        d = Path(ffmpeg_path).resolve()
        for name in ("ffmpeg.exe", "ffmpeg"):
            if (d / name).exists():
                return d
        bin_dir = d / "bin"
        if bin_dir.is_dir() and (bin_dir / "ffmpeg.exe").exists():
            return bin_dir
    return _search_common_ffmpeg_locations()


def require_ffmpeg() -> None:
    """
    Ensure ffmpeg is available (required by pydub for MP3).
    If FFMPEG_PATH is set, prepend it to PATH so subprocess finds it.
    """
    bin_dir = _find_ffmpeg_bin()
    if bin_dir is not None:
        path_str = str(bin_dir)
        if path_str not in os.environ.get("PATH", "").split(os.pathsep)[:1]:
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        return
    # Build helpful error
    hint = (
        "Add the folder containing ffmpeg.exe to PATH (e.g. ...\\ffmpeg-8.0.1-essentials_build\\bin), "
        "or set environment variable FFMPEG_PATH to that folder. Then restart the server."
    )
    raise RuntimeError(
        "ffmpeg is required for audio generation but was not found. " + hint
    )


def build_ssml(text: str, pause_after_full_stop_ms: int = 300) -> str:
    """
    Build minimal SSML for pauses and emphasis (for engines that support it, e.g. ElevenLabs).
    Inserts a short break after sentence-ending punctuation.
    """
    import re
    # Insert break after . ! ?
    pattern = r"([.!?])\s+"
    replacement = r"\1<break time=\"" + str(pause_after_full_stop_ms) + "ms\"/> "
    with_breaks = re.sub(pattern, replacement, text)
    return f"<speak>{with_breaks.strip()}</speak>"
