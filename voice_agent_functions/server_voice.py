"""Server-side voice I/O — works without any API keys.

Provides two things the browser can't always do reliably:

* **TTS** via `edge-tts` (Microsoft neural voices, free, no key). Telugu
  voices: `te-IN-ShrutiNeural` (female, default) and
  `te-IN-MohanNeural` (male).
* **STT** via `vosk` with the small Telugu model
  (`models/vosk-model-small-te-0.42`). Fully offline.

Both are optional dependencies — if they're missing, the functions
raise a clear error and the caller falls back to browser APIs.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("ruchi.server_voice")

_HERE = Path(__file__).resolve().parent
# Models live next to the package (project_root/models/...).
_PROJECT_ROOT = _HERE.parent
VOSK_MODEL_DIR = _PROJECT_ROOT / "models" / "vosk-model-small-te-0.42"

# Telugu neural voices from edge-tts.
TELUGU_VOICES = ("te-IN-ShrutiNeural", "te-IN-MohanNeural")
DEFAULT_TTS_VOICE = "te-IN-ShrutiNeural"

# A single shared AsyncClient keeps connections (and TLS handshakes) warm
# across turns — creating one per request added ~0.1-0.3s each call.
_http: httpx.AsyncClient | None = None


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _http


async def aclose() -> None:
    """Close the shared client (called on app shutdown)."""
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


# ============================================================
# TTS — ElevenLabs (preferred when key is set)
# ============================================================
def elevenlabs_configured() -> bool:
    return bool(os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY"))


async def synthesize_elevenlabs(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> bytes:
    """Return MP3 bytes from ElevenLabs.

    Defaults to the model in ``ELEVEN_MODEL_ID`` (``eleven_v3_conversational``
    per .env) and the voice in ``ELEVEN_VOICE_ID``.
    """
    key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    vid = voice_id or os.getenv("ELEVEN_VOICE_ID")
    model = model_id or os.getenv("ELEVEN_MODEL_ID", "eleven_v3_conversational")
    if not key:
        raise RuntimeError("ELEVEN_API_KEY is not set")
    if not vid:
        raise RuntimeError("ELEVEN_VOICE_ID is not set")
    if not text.strip():
        raise ValueError("text is empty")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{vid}"
        "?output_format=mp3_44100_128"
    )
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text.strip()[:2000],
        "model_id": model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    r = await _client().post(url, headers=headers, json=payload)
    if r.status_code >= 400:
        raise RuntimeError(f"ElevenLabs {r.status_code}: {r.text[:300]}")
    if not r.content:
        raise RuntimeError("ElevenLabs returned no audio")
    return bytes(r.content)


async def stream_elevenlabs(
    text: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    output_format: str = "pcm_16000",
):
    """Yield raw PCM chunks from ElevenLabs' streaming endpoint.

    Streaming lets the browser start playing after the first chunk
    (~1s) instead of waiting for the whole clip (~3s). Yields
    ``bytes`` chunks of signed 16-bit little-endian mono PCM at the
    sample rate encoded in ``output_format`` (16000 by default).
    """
    key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    vid = voice_id or os.getenv("ELEVEN_VOICE_ID")
    model = model_id or os.getenv("ELEVEN_MODEL_ID", "eleven_v3_conversational")
    if not key:
        raise RuntimeError("ELEVEN_API_KEY is not set")
    if not vid:
        raise RuntimeError("ELEVEN_VOICE_ID is not set")
    if not text.strip():
        raise ValueError("text is empty")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{vid}/stream"
        f"?output_format={output_format}"
    )
    headers = {
        "xi-api-key": key,
        "Content-Type": "application/json",
        "Accept": "audio/pcm",
    }
    payload = {
        "text": text.strip()[:2000],
        "model_id": model,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    async with _client().stream("POST", url, headers=headers, json=payload) as r:
        if r.status_code >= 400:
            body = (await r.aread()).decode(errors="replace")[:300]
            raise RuntimeError(f"ElevenLabs stream {r.status_code}: {body}")
        async for chunk in r.aiter_bytes():
            if chunk:
                yield chunk


def pcm_to_wav(
    pcm: bytes, sample_rate: int = 16000, channels: int = 1, sampwidth: int = 2
) -> bytes:
    """Wrap raw PCM bytes in a WAV container (for the session recording)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


# ============================================================
# TTS text formatting — slow, clear lists
# ============================================================
_BULLET_RE = re.compile(r"^\s*[-•*]\s+(.*)$")
_LIST_LABEL_RE = re.compile(r"^(?P<label>[^:]{1,40}):\s*(?P<rest>.+)$")

# ElevenLabs v4 audio tags, e.g. [warm], [warm, cheerful], [laughs].
_EMOTION_TAG_RE = re.compile(r"\[\s*[A-Za-z]+(?:\s*,\s*[A-Za-z]+)*\s*\]")


def strip_emotion_tags(text: str) -> str:
    """Remove ElevenLabs emotion tags from text for display/logging.

    The tags are kept for TTS (they control delivery) but must not appear
    in the UI, transcripts, or stored memory.
    """
    if not text:
        return text
    cleaned = _EMOTION_TAG_RE.sub(" ", text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"^[ \t]+", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


def format_for_tts(text: str) -> str:
    """Insert natural pauses into ingredient/utensil lists.

    Bullet lists become ellipsis-separated items, and inline lists after a
    short label (``Ingredients: a, b, c``) get their commas turned into
    ellipses. Normal conversation is returned unchanged so it isn't slowed.
    """
    if not text:
        return text
    lines = text.splitlines()

    # 1) Bullet / dash lists → "item… item… and item." while KEEPING any
    #    surrounding lines (e.g. a leading [warm] tag or an intro sentence).
    if any(_BULLET_RE.match(l) for l in lines):
        out: list[str] = []
        items: list[str] = []

        def flush() -> None:
            if items:
                out.append("… ".join(items) + ".")
                items.clear()

        for l in lines:
            m = _BULLET_RE.match(l)
            if m:
                item = m.group(1).strip().rstrip(",;.")
                if item:
                    items.append(item)
            else:
                flush()
                if l.strip():
                    out.append(l.strip())
        flush()
        return "\n".join(out)

    # 2) "Label: a, b, c" inline lists with 3+ short comma items.
    out: list[str] = []
    for l in lines:
        m = _LIST_LABEL_RE.match(l.strip())
        if m and m.group("rest").count(",") >= 2 and len(m.group("rest")) < 400:
            parts = [
                p.strip().rstrip(",;.")
                for p in re.split(r",\s*", m.group("rest"))
                if p.strip()
            ]
            if len(parts) >= 3:
                l = f"{m.group('label')}: " + "… ".join(parts) + "."
        out.append(l)
    return "\n".join(out)


# ============================================================
# STT — Sarvam (preferred when key is set)
# ============================================================
def sarvam_configured() -> bool:
    return bool(os.getenv("SARVAM_API_KEY"))


async def transcribe_sarvam(
    audio_bytes: bytes,
    filename: str = "audio.webm",
    content_type: str = "audio/webm",
    language_code: Optional[str] = None,
) -> str:
    """Transcribe with Sarvam saaras:v3 (Telugu by default)."""
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is not set")
    if not audio_bytes:
        raise ValueError("audio is empty")
    lang = language_code or os.getenv("SARVAM_LANGUAGE", "te-IN")
    model = os.getenv("SARVAM_MODEL", "saaras:v3")

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {"api-subscription-key": key}
    files = {"file": (filename, io.BytesIO(audio_bytes), content_type)}
    data = {"model": model, "language_code": lang}
    r = await _client().post(url, headers=headers, files=files, data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"Sarvam {r.status_code}: {r.text[:300]}")
    body = r.json()
    for k in ("transcript", "text", "transcription"):
        if body.get(k):
            return str(body[k])
    nested = body.get("data") or {}
    if isinstance(nested, dict):
        for k in ("transcript", "text"):
            if nested.get(k):
                return str(nested[k])
    return ""


# ============================================================
# TTS — edge-tts (free fallback)
# ============================================================
async def synthesize_edge_tts(
    text: str,
    voice: str = DEFAULT_TTS_VOICE,
    rate: str = "-5%",
) -> bytes:
    """Return MP3 bytes for ``text`` spoken in Telugu.

    Raises ``RuntimeError`` if edge-tts isn't installed or the network
    call fails (edge-tts uses Microsoft's public endpoint).
    """
    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("edge-tts is not installed (pip install edge-tts)") from exc

    if not text.strip():
        raise ValueError("text is empty")
    if voice not in TELUGU_VOICES:
        voice = DEFAULT_TTS_VOICE

    comm = edge_tts.Communicate(text.strip()[:1500], voice, rate=rate)
    audio = bytearray()
    async for chunk in comm.stream():
        if chunk.get("type") == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(audio)


# ============================================================
# STT — vosk (offline)
# ============================================================
_vosk_model = None  # cached KaldiRecognizer model


def _get_vosk_model():
    """Load and cache the Vosk Telugu model. Raises RuntimeError if
    vosk isn't installed or the model directory is missing."""
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model
    try:
        from vosk import Model, SetLogLevel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("vosk is not installed (pip install vosk)") from exc
    if not VOSK_MODEL_DIR.is_dir():
        raise RuntimeError(
            f"Vosk Telugu model not found at {VOSK_MODEL_DIR}. "
            "Download it from https://alphacephei.com/vosk/models "
            "(vosk-model-small-te-0.42) and unzip into models/."
        )
    SetLogLevel(-1)  # quiet
    logger.info("loading Vosk model from %s", VOSK_MODEL_DIR)
    _vosk_model = Model(str(VOSK_MODEL_DIR))
    return _vosk_model


def _to_wav16k_mono(audio_bytes: bytes, src_name: str = "audio.webm") -> bytes:
    """Convert arbitrary audio to 16 kHz mono 16-bit PCM WAV using
    ffmpeg. Returns WAV bytes. Raises RuntimeError if ffmpeg is
    missing or conversion fails."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — needed to convert audio for Vosk")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / src_name
        dst = Path(td) / "out.wav"
        src.write_bytes(audio_bytes)
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(src),
            "-ar", "16000", "-ac", "1", "-f", "wav",
            str(dst),
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:300]}"
            )
        return dst.read_bytes()


def transcribe_vosk(audio_bytes: bytes, src_name: str = "audio.webm") -> str:
    """Transcribe ``audio_bytes`` with the Vosk Telugu model.

    Accepts any format ffmpeg can read (webm/ogg/mp3/wav).
    Returns the transcript (may be empty).
    """
    if not audio_bytes:
        raise ValueError("audio is empty")
    wav_bytes = _to_wav16k_mono(audio_bytes, src_name)
    model = _get_vosk_model()

    from vosk import KaldiRecognizer
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        wav_path = f.name
    try:
        with wave.open(wav_path, "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                raise RuntimeError("expected 16-bit mono WAV after conversion")
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                rec.AcceptWaveform(data)
        result = json.loads(rec.FinalResult())
        return (result.get("text") or "").strip()
    finally:
        Path(wav_path).unlink(missing_ok=True)


# ============================================================
# Capability probe (used by /api/voice/capabilities)
# ============================================================
def capabilities() -> dict:
    """Report which server-side voice pieces are available.

    The frontend uses this to decide whether to hit the server or fall
    back to browser APIs.
    """
    caps = {
        # LLM
        "gemini": bool(os.getenv("GOOGLE_API_KEY")),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        # Preferred (real) services — enabled by keys in .env
        "elevenlabs": elevenlabs_configured(),
        "elevenlabs_model": os.getenv("ELEVEN_MODEL_ID", "eleven_v3_conversational"),
        "elevenlabs_voice": os.getenv("ELEVEN_VOICE_ID", ""),
        # Sarvam STT (preferred when key is set)
        "sarvam": sarvam_configured(),
        "sarvam_model": os.getenv("SARVAM_MODEL", "saaras:v3"),
        # LiveKit (optional realtime path)
        "livekit": bool(
            os.getenv("LIVEKIT_URL") and os.getenv("LIVEKIT_API_KEY") and os.getenv("LIVEKIT_API_SECRET")
        ),
        # Free fallbacks
        "edge_tts": False,
        "edge_tts_voices": list(TELUGU_VOICES),
        "vosk": False,
        "vosk_model": VOSK_MODEL_DIR.is_dir(),
        "ffmpeg": bool(shutil.which("ffmpeg")),
    }
    try:
        import edge_tts  # noqa: F401
        caps["edge_tts"] = True
    except ImportError:
        pass
    try:
        import vosk  # noqa: F401
        caps["vosk"] = True
    except ImportError:
        pass
    return caps


__all__ = [
    "TELUGU_VOICES",
    "DEFAULT_TTS_VOICE",
    "VOSK_MODEL_DIR",
    "synthesize_elevenlabs",
    "stream_elevenlabs",
    "pcm_to_wav",
    "format_for_tts",
    "strip_emotion_tags",
    "aclose",
    "synthesize_edge_tts",
    "transcribe_sarvam",
    "transcribe_vosk",
    "elevenlabs_configured",
    "sarvam_configured",
    "capabilities",
]
