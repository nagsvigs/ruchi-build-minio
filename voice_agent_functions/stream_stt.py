"""Realtime streaming STT via Sarvam's WebSocket endpoint.

Proxies the browser's microphone PCM (16 kHz mono linear16) to Sarvam's
``saaras:v3-realtime`` WebSocket and relays transcripts + VAD (turn)
events back to the browser. The Sarvam API key never leaves the server.

Turn detection is done by Sarvam's server-side VAD (``endpointing=vad``),
tunable with ``SARVAM_SILENCE_MS`` / ``SARVAM_VAD_THRESHOLD`` /
``SARVAM_MIN_SPEECH_MS``. That removes the browser's fixed silence guess
and the batch ``/api/voice/stt`` upload, so replies start sooner.

Server events relayed to the browser::

    {"event": "session.begin", ...}
    {"event": "vad.speech_start"}          -> barge-in cue
    {"event": "vad.speech_end"}
    {"event": "transcript.partial", "text": "..."}
    {"event": "transcript.final",   "text": "..."}   -> run the turn
    {"event": "error", "message": "..."}
"""
from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, AsyncIterator

logger = logging.getLogger("ruchi.stream_stt")

SARVAM_WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"


def configured() -> bool:
    return bool(os.getenv("SARVAM_API_KEY"))


def _query() -> str:
    params = {
        "language_code": os.getenv("SARVAM_LANGUAGE", "te-IN"),
        "model": os.getenv("SARVAM_REALTIME_MODEL", "saaras:v3-realtime"),
        "stream_type": os.getenv("SARVAM_STREAM_TYPE", "fast"),
        "endpointing": os.getenv("SARVAM_ENDPOINTING", "vad"),
        "encoding": "linear16",
        "sample_rate": "16000",
        "threshold": os.getenv("SARVAM_VAD_THRESHOLD", "0.3"),
        "silence_duration_ms": os.getenv("SARVAM_SILENCE_MS", "500"),
        "min_speech_duration_ms": os.getenv("SARVAM_MIN_SPEECH_MS", "250"),
    }
    return "&".join(f"{k}={v}" for k, v in params.items())


class SarvamStream:
    """One live streaming-STT session to Sarvam."""

    def __init__(self) -> None:
        self._ws: Any = None
        self._key = os.getenv("SARVAM_API_KEY", "")

    async def connect(self) -> None:
        import websockets

        url = f"{SARVAM_WS_URL}?{_query()}"
        headers = {"api-subscription-key": self._key}
        # Sarvam doesn't answer WebSocket-level control pings, so the
        # library's automatic ping would time out and close the socket
        # (~20s) with "keepalive ping timeout". Audio frames keep the
        # connection alive at the application level, so disable it.
        kwargs = dict(ping_interval=None, max_size=None)
        try:
            self._ws = await websockets.connect(
                url, additional_headers=headers, **kwargs
            )
        except TypeError:  # websockets < 14 uses extra_headers
            self._ws = await websockets.connect(url, extra_headers=headers, **kwargs)
        logger.info("Sarvam realtime STT connected")

    async def send_audio(self, pcm: bytes) -> None:
        if self._ws is None or not pcm:
            return
        payload = {
            "event": "audio_input",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }
        await self._ws.send(json.dumps(payload))

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        if self._ws is None:
            return
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                yield json.loads(raw)
            except (ValueError, TypeError):
                continue

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 - best-effort close
                pass
            self._ws = None


__all__ = ["configured", "SarvamStream", "SARVAM_WS_URL"]
