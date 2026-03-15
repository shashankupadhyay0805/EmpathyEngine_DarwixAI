"""
Empathy Engine: FastAPI application and CLI entrypoint.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from app.config import PROJECT_ROOT
from app.emotion_detector import detect_emotion
from app.utils import get_logger, generate_audio_filename
from app.voice_mapper import get_speech_params
from app.tts_engine import synthesize_speech, save_audio

logger = get_logger(__name__)


def run_pipeline(
    text: str,
    output_path: Optional[Path] = None,
    intensity: float = 1.0,
    lang: str = "en",
    audio_format: str = "mp3",
) -> tuple[str, Path]:
    """
    Run the full pipeline: detect emotion, map to params, synthesize, save.

    Returns:
        Tuple of (emotion_category, path_to_audio_file).
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")
    fmt = "mp3" if audio_format.lower() != "wav" else "wav"

    emotion_result = detect_emotion(text)
    params = get_speech_params(emotion_result, intensity=intensity)
    logger.info("Emotion: %s -> params: %s", emotion_result.emotion, params)

    audio = synthesize_speech(text, params, lang=lang)
    path = output_path or generate_audio_filename(fmt)
    save_audio(audio, path, format=fmt)

    return emotion_result.emotion, path


# --- Web demo HTML (inline fallback) ---

def _default_demo_html() -> str:
    """Default demo page HTML if static file is missing."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Empathy Engine</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 560px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    p { color: #555; margin-bottom: 1.5rem; }
    textarea { width: 100%; min-height: 100px; padding: 0.75rem; border: 1px solid #ccc; border-radius: 6px; resize: vertical; }
    button { margin-top: 0.75rem; padding: 0.6rem 1.2rem; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #result { margin-top: 1rem; padding: 0.75rem; background: #f3f4f6; border-radius: 6px; }
    #result .emotion { font-weight: 600; }
    audio { width: 100%; margin-top: 0.5rem; }
    .error { color: #b91c1c; }
  </style>
</head>
<body>
  <h1>Empathy Engine</h1>
  <p>Enter text to generate emotionally expressive speech.</p>
  <textarea id="text" placeholder="e.g. Your order has been successfully delivered!">Your order has been successfully delivered!</textarea>
  <br>
  <button id="btn">Generate voice</button>
  <div id="result"></div>
  <script>
    const text = document.getElementById('text');
    const btn = document.getElementById('btn');
    const result = document.getElementById('result');
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      result.innerHTML = '';
      try {
        const r = await fetch('/generate-voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text.value.trim() })
        });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'Request failed');
        result.innerHTML = '<span class="emotion">Emotion: ' + data.emotion + '</span><br><audio controls src="/audio/' + data.audio_file.split('/').pop() + '"></audio>';
      } catch (e) {
        result.innerHTML = '<span class="error">' + e.message + '</span>';
      }
      btn.disabled = false;
    });
  </script>
</body>
</html>"""


# --- FastAPI app ---

def create_app():
    """Create FastAPI application (used by uvicorn and tests)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel, Field, field_validator

    app = FastAPI(
        title="Empathy Engine",
        description="Convert text to emotionally expressive speech.",
        version="1.0.0",
    )

    class GenerateRequest(BaseModel):
        text: str = Field(..., min_length=1, description="Input text to convert to speech")
        format: str = Field(default="mp3", description="Output format: 'mp3' or 'wav'")

        @field_validator("format")
        @classmethod
        def format_must_be_mp3_or_wav(cls, v: str) -> str:
            if v and v.strip().lower() in ("mp3", "wav"):
                return v.strip().lower()
            return "mp3"

    class GenerateResponse(BaseModel):
        emotion: str
        audio_file: str

    @app.post("/generate-voice", response_model=GenerateResponse)
    def generate_voice(body: GenerateRequest):
        """Generate expressive speech from text. Returns playable .mp3 or .wav."""
        try:
            emotion, path = run_pipeline(
                body.text,
                output_path=None,
                intensity=1.0,
                lang="en",
                audio_format=body.format.strip().lower() or "mp3",
            )
            try:
                rel_path = path.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = path
            return GenerateResponse(
                emotion=emotion,
                audio_file=str(rel_path).replace("\\", "/"),
            )
        except (RuntimeError, FileNotFoundError, OSError) as e:
            msg = str(e)
            if "ffmpeg" in msg.lower() or "cannot find the file" in msg.lower():
                raise HTTPException(
                    status_code=503,
                    detail="Audio generation requires ffmpeg. Add the folder containing ffmpeg.exe to your PATH (e.g. ...\\ffmpeg-8.0.1-essentials_build\\bin), or set FFMPEG_PATH to that folder and restart the server.",
                )
            raise HTTPException(status_code=500, detail=msg)
        except Exception as e:
            logger.exception("Generate voice failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/", response_class=HTMLResponse)
    def demo_page():
        """Serve the web demo HTML page."""
        html_path = PROJECT_ROOT / "app" / "static" / "demo.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
        return HTMLResponse(content=_default_demo_html(), status_code=200)

    @app.get("/audio/{filename:path}")
    def serve_audio(filename: str):
        """Serve a generated audio file by path under outputs/generated_audio."""
        path = PROJECT_ROOT / "outputs" / "generated_audio" / filename
        outputs_dir = (PROJECT_ROOT / "outputs").resolve()
        if not str(path.resolve()).startswith(str(outputs_dir)):
            raise HTTPException(status_code=403, detail="Invalid path")
        if not path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        media_type = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
        return FileResponse(path, media_type=media_type)

    return app


app = create_app()


# --- CLI ---

def cli() -> None:
    """Run the Empathy Engine from the command line."""
    parser = argparse.ArgumentParser(description="Empathy Engine — text to expressive speech")
    parser.add_argument("--text", "-t", help="Text to convert to speech (required unless --serve)")
    parser.add_argument("--output", "-o", type=Path, help="Output audio file path")
    parser.add_argument("--format", "-f", choices=("mp3", "wav"), default="mp3", help="Output format: mp3 or wav")
    parser.add_argument("--intensity", "-i", type=float, default=1.0, help="Emotion intensity 0.0–1.0")
    parser.add_argument("--lang", "-l", default="en", help="Language code for TTS")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server instead of one-shot")
    parser.add_argument("--host", default="0.0.0.0", help="Host for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=False,
        )
        return

    if not args.text:
        parser.error("--text is required when not using --serve")

    try:
        emotion, path = run_pipeline(
            args.text,
            output_path=args.output,
            intensity=args.intensity,
            lang=args.lang,
            audio_format=args.format,
        )
        print(f"Emotion: {emotion}")
        print(f"Audio: {path}")
    except Exception as e:
        logger.exception("CLI failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
