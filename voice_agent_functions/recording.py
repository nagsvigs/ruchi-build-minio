"""Per-session audio recording: user mic + Ruchi's voice -> one MP3.

Both audio streams already pass through this FastAPI server:

* the user's microphone audio is POSTed to ``/api/voice/stt``
* Ruchi's spoken reply is produced by ``/api/voice/tts``

So we simply append each blob to a per-session folder as an ordered
segment, and on demand concatenate + transcode everything to a single
MP3 with ffmpeg.

Layout::

    recordings/
        <session_id>.mp3                 # final mix, built on demand
        <session_id>/
            segments/
                00000000000000000001_user.webm
                00000000000000000002_ruchi.mp3
                ...

Segments are named with a nanosecond timestamp so their lexicographic
order is the real chronological order of the conversation, regardless
of which stream (mic or TTS) produced them.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("ruchi.recording")

_HERE = Path(__file__).resolve().parent
DEFAULT_DIR = _HERE.parent / "recordings"

# Serializes segment writes and ffmpeg runs.
_LOCK = threading.Lock()

# Audio container per stream kind.
_EXT_BY_KIND = {"user": "webm", "ruchi": "mp3"}

# Seconds; generous so long recordings still finalize.
_FFMPEG_TIMEOUT = 600


def enabled() -> bool:
    """Recording is on by default; set ``RECORDING_ENABLED=0`` to disable."""
    return os.getenv("RECORDING_ENABLED", "1").lower() not in ("0", "false", "no", "off", "")


def recordings_dir() -> Path:
    d = Path(os.getenv("RECORDINGS_DIR", str(DEFAULT_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_id(session_id: str) -> str:
    keep = "".join(c for c in (session_id or "") if c.isalnum() or c in "-_")
    return keep[:64] or "unknown"


def segments_dir(session_id: str) -> Path:
    d = recordings_dir() / _safe_id(session_id) / "segments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def mp3_path(session_id: str) -> Path:
    return recordings_dir() / f"{_safe_id(session_id)}.mp3"


def add_segment(
    session_id: str | None, kind: str, data: bytes, ext: str | None = None
) -> Path | None:
    """Append one audio blob for ``session_id``.

    ``kind`` is ``"user"`` (mic) or ``"ruchi"`` (TTS). ``ext`` overrides
    the default container for the kind (e.g. ``"wav"`` when the TTS was
    streamed as PCM and wrapped in a WAV header). Returns the written
    path, or ``None`` when recording is disabled / there's nothing to save.
    """
    if not enabled() or not session_id or not data:
        return None
    ext = ext or _EXT_BY_KIND.get(kind, "bin")
    # time_ns prefix -> chronological ordering across both streams.
    name = f"{time.time_ns():020d}_{kind}.{ext}"
    path = segments_dir(session_id) / name
    with _LOCK:
        path.write_bytes(data)
    return path


def _has_audio(path: Path) -> bool:
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        return "audio" in r.stdout
    except Exception:
        return False


def _segment_files(session_id: str) -> list[Path]:
    d = segments_dir(session_id)
    return sorted(p for p in d.iterdir() if p.is_file())


def finalize(session_id: str) -> Path | None:
    """Concatenate every segment into ``recordings/<session_id>.mp3``.

    Safe to call repeatedly: if no new segments arrived since the last
    build, the existing MP3 is returned. Returns ``None`` when there is
    nothing to record or ffmpeg fails.
    """
    if not enabled() or not session_id:
        return None

    segs = _segment_files(session_id)
    if not segs:
        return None

    out = mp3_path(session_id)
    work = recordings_dir() / _safe_id(session_id) / "work"

    with _LOCK:
        # Reuse the existing MP3 if it's newer than every segment.
        if out.exists():
            newest = max(p.stat().st_mtime for p in segs)
            if out.stat().st_mtime >= newest:
                return out

        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        # 1) Normalize each segment to a common WAV (mixed webm/opus + mp3).
        wavs: list[Path] = []
        for i, seg in enumerate(segs):
            if not _has_audio(seg):
                continue
            wav = work / f"{i:05d}.wav"
            r = subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-i", str(seg),
                    "-vn", "-ar", "44100", "-ac", "1",
                    "-c:a", "pcm_s16le", str(wav),
                ],
                capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
            )
            if r.returncode == 0 and wav.exists():
                wavs.append(wav)
            else:
                log.warning("recording: skipping %s: %s", seg.name, (r.stderr or "")[:200])

        if not wavs:
            return None

        # 2) Concat the normalized WAVs and encode a single MP3.
        list_file = work / "concat.txt"
        list_file.write_text(
            "".join(f"file '{w.as_posix()}'\n" for w in wavs), encoding="utf-8"
        )
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", str(list_file), "-c:a", "libmp3lame", "-q:a", "4", str(out),
            ],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT,
        )
        if r.returncode != 0 or not out.exists():
            log.error("recording: ffmpeg concat failed: %s", (r.stderr or "")[:400])
            return None

    log.info("recording: wrote %s (%d bytes)", out.name, out.stat().st_size)
    return out


def list_recordings() -> list[dict]:
    d = recordings_dir()
    items = []
    for p in sorted(d.glob("*.mp3"), key=lambda x: x.stat().st_mtime, reverse=True):
        st = p.stat()
        items.append(
            {
                "session_id": p.stem,
                "file": p.name,
                "bytes": st.st_size,
                "modified": int(st.st_mtime),
            }
        )
    return items


def delete(session_id: str) -> None:
    d = recordings_dir() / _safe_id(session_id)
    shutil.rmtree(d, ignore_errors=True)
    mp3_path(session_id).unlink(missing_ok=True)


__all__ = [
    "enabled",
    "recordings_dir",
    "segments_dir",
    "mp3_path",
    "add_segment",
    "finalize",
    "list_recordings",
    "delete",
]
