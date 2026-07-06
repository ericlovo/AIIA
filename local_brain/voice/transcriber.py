"""
Local voice transcription via faster-whisper (runs on Apple Silicon CPU/Metal).

Model is loaded once at startup and reused. Uses "base" by default (~150MB),
switchable to "small" or "medium" via WHISPER_MODEL env var.
"""

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger("aiia.voice")

_model = None
_model_size = os.getenv("WHISPER_MODEL", "base")


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        from faster_whisper import WhisperModel

        logger.info(f"Loading Whisper {_model_size} model...")
        _model = WhisperModel(_model_size, device="cpu", compute_type="int8")
        logger.info("Whisper ready")
    except ImportError:
        logger.warning("faster-whisper not installed — voice transcription unavailable")
        _model = None
    return _model


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.webm") -> dict:
    model = _load_model()
    if model is None:
        return {"text": "", "error": "faster-whisper not installed"}

    suffix = Path(filename).suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = model.transcribe(tmp_path, beam_size=5, language="en")
        text = " ".join(s.text.strip() for s in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 1),
        }
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"text": "", "error": str(e)}
    finally:
        Path(tmp_path).unlink(missing_ok=True)
