"""Standalone FastAPI entry point for voice_agent_functions.

Lets you run the demo + 12 function-calling tools from this folder
without needing the full ruchi-basicfullstack backend:

    cd ~/Projects/voice_agent_functions
    ./run.sh
    # then open http://127.0.0.1:8000/demo/

Serving the demo over http://127.0.0.1 (a "secure context") is
important — browsers only allow microphone access and Web Speech
recognition on localhost or HTTPS. Opening demo/index.html directly
as a file:// URL blocks the mic.

Endpoints:
  GET  /                                  server info
  GET  /api/health                        status
  GET  /api/voice/capabilities            which server-side voice bits exist
  POST /api/voice/tts   {text, voice?}    → MP3 (edge-tts Telugu neural)
  POST /api/voice/stt   multipart file    → {transcript} (Vosk Telugu, offline)
  GET  /api/voice/voices                  list available Telugu voices
  + all /api/voice-agent/* tool endpoints
  + /demo/                                the interactive demo (static)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env BEFORE importing modules that read env vars at import time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from voice_agent_functions import recording, server_voice, stream_stt
from voice_agent_functions import recipe_catalog
from voice_agent_functions.routes import router as voice_agent_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
# httpx logs full request URLs at INFO, which would leak API keys
# (Gemini/ElevenLabs) into the server log — keep it at WARNING.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ruchi.standalone")

# Small LRU cache of synthesized audio for fixed phrases (greeting, the
# fixed stage questions). Keyed by (kind, text) where kind is "pcm" or
# "mp3". Repeat phrases are then served with no API round-trip.
_TTS_CACHE: "OrderedDict[tuple[str, str], bytes]" = OrderedDict()
_TTS_CACHE_MAX = 64


def _cache_get(kind: str, text: str) -> Optional[bytes]:
    key = (kind, text)
    data = _TTS_CACHE.get(key)
    if data is not None:
        _TTS_CACHE.move_to_end(key)
    return data


def _cache_put(kind: str, text: str, data: bytes) -> None:
    if not data or len(data) > 4_000_000:
        return
    key = (kind, text)
    _TTS_CACHE[key] = data
    _TTS_CACHE.move_to_end(key)
    while len(_TTS_CACHE) > _TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)

_HERE = Path(__file__).resolve().parent
DEMO_DIR = _HERE / "demo"

app = FastAPI(
    title="Ruchi — voice_agent_functions standalone",
    version="2.1.0",
    description=(
        "Standalone FastAPI server exposing the 12 function-calling tools "
        "from ruchi-voice-agent PR #1, plus free Telugu TTS (edge-tts) and "
        "offline Telugu STT (Vosk). Open /demo/ to interact."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-TTS-Provider", "X-Sample-Rate"],
)

app.include_router(voice_agent_router)


# ---------------------------------------------------------------- models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    # Optional: when set, the synthesized audio is appended to that
    # session's recording (see voice_agent_functions/recording.py).
    session_id: Optional[str] = None
    # When true and ElevenLabs is configured, stream PCM chunks so the
    # browser can start playing after the first chunk (~1s) instead of
    # waiting for the whole clip (~3s).
    stream: bool = False


# ---------------------------------------------------------------- info
@app.get("/api/health")
async def health():
    caps = server_voice.capabilities()
    return {
        "status": "ok",
        "name": "voice_agent_functions standalone",
        "voice": caps,
        "demo": "/demo/",
        "endpoints": [
            "/api/voice-agent/sessions",
            "/api/voice-agent/sessions/{sid}",
            "/api/voice-agent/sessions/{sid}/turn",
            "/api/voice-agent/sessions/{sid}/call",
            "/api/voice-agent/sessions/{sid}/recipe",
            "/api/voice-agent/sessions/{sid}/save_answer",
            "/api/voice-agent/sessions/{sid}/recording",
            "/api/voice-agent/tools",
            "/api/voice-agent/prompts",
            "/api/voice-agent/transcript",
            "/api/voice-agent/recordings",
            "/api/voice/tts",
            "/api/voice/stt",
            "/api/voice/voices",
            "/api/voice/capabilities",
            "/demo/",
            "/docs",
        ],
    }


@app.get("/")
async def root():
    return {
        "name": "Ruchi — voice_agent_functions standalone",
        "demo": "http://127.0.0.1:8000/demo/  (open this, not the file:// path)",
        "docs": "/docs",
        "health": "/api/health",
    }


# ---------------------------------------------------------------- voice
@app.get("/api/voice/capabilities")
async def voice_capabilities():
    return server_voice.capabilities()


# ---------------------------------------------------------------- recipes
@app.get("/api/recipes")
async def recipes_list():
    """List the bundled recipe catalog (id, name, category, region)."""
    cat = recipe_catalog.get_catalog()
    return {"count": len(cat), "recipes": cat.list_recipes()}


@app.get("/api/recipes/search")
async def recipes_search(q: str):
    """Resolve a spoken dish name to a recipe (or candidates)."""
    return recipe_catalog.resolve(q)


@app.get("/api/recipes/{recipe_id}")
async def recipes_get(recipe_id: str):
    cat = recipe_catalog.get_catalog()
    entry = cat.get(recipe_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"recipe {recipe_id!r} not found")
    return {
        "id": entry.id,
        "name": entry.name,
        "region": entry.region,
        "category": entry.category,
        "recipe": entry.recipe,
    }


@app.get("/api/voice/voices")
async def voice_voices():
    return {"voices": list(server_voice.TELUGU_VOICES), "default": server_voice.DEFAULT_TTS_VOICE}


@app.post("/api/voice/tts")
async def voice_tts(req: TTSRequest):
    """Synthesize Telugu speech.

    Preference order:
      1. ElevenLabs (``ELEVEN_MODEL_ID``, default ``eleven_v3_conversational``)
      2. edge-tts (free, no key)
      3. browser speechSynthesis (frontend fallback)
    Returns audio/mpeg, or a JSON ``{fallback}`` object.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")

    # Serve repeated fixed phrases (greeting, stage questions) from cache
    # with no API round-trip.
    if req.stream:
        cached = _cache_get("pcm", req.text)
        if cached:
            return Response(
                content=cached,
                media_type="audio/pcm",
                headers={
                    "X-TTS-Provider": "cache",
                    "X-Sample-Rate": "16000",
                    "Cache-Control": "no-store",
                },
            )
    else:
        cached = _cache_get("mp3", req.text)
        if cached:
            return Response(
                content=cached,
                media_type="audio/mpeg",
                headers={"X-TTS-Provider": "cache"},
            )

    # 0) Streaming ElevenLabs — lowest latency. Returns raw 16 kHz mono
    #    PCM; the browser plays it incrementally via the Web Audio API.
    if req.stream and server_voice.elevenlabs_configured():
        return StreamingResponse(
            _stream_tts(req.text, req.voice, req.session_id),
            media_type="audio/pcm",
            headers={
                "X-TTS-Provider": "elevenlabs",
                "X-Sample-Rate": "16000",
                "Cache-Control": "no-store",
            },
        )

    # Slow down ingredient/utensil lists for clarity (normal text unchanged).
    speech = server_voice.format_for_tts(req.text)

    # 1) ElevenLabs
    if server_voice.elevenlabs_configured():
        try:
            audio = await server_voice.synthesize_elevenlabs(speech, req.voice)
            _cache_put("mp3", req.text, audio)
            recording.add_segment(req.session_id, "ruchi", audio)
            return Response(
                content=audio,
                media_type="audio/mpeg",
                headers={"X-TTS-Provider": "elevenlabs"},
            )
        except Exception as exc:
            log.warning("ElevenLabs TTS failed (%s) — falling back to edge-tts", exc)

    # 2) edge-tts
    if server_voice.capabilities().get("edge_tts"):
        try:
            audio = await server_voice.synthesize_edge_tts(
                speech, req.voice or server_voice.DEFAULT_TTS_VOICE
            )
            _cache_put("mp3", req.text, audio)
            recording.add_segment(req.session_id, "ruchi", audio)
            return Response(
                content=audio,
                media_type="audio/mpeg",
                headers={"X-TTS-Provider": "edge-tts"},
            )
        except Exception as exc:
            log.warning("edge-tts failed: %s", exc)

    # 3) Browser
    return JSONResponse(
        status_code=200,
        content={"fallback": "browser", "reason": "no server TTS available", "text": req.text},
    )


async def _stream_tts(text: str, voice: Optional[str], session_id: Optional[str]):
    """Stream ElevenLabs PCM to the client, saving it to the session
    recording as a WAV segment once complete."""
    speech = server_voice.format_for_tts(text)
    pcm = bytearray()
    try:
        async for chunk in server_voice.stream_elevenlabs(speech, voice):
            pcm.extend(chunk)
            yield chunk
    except Exception as exc:  # client already has headers; just stop
        log.warning("streaming TTS failed: %s", exc)
    finally:
        if pcm:
            _cache_put("pcm", text, bytes(pcm))
            if session_id:
                try:
                    recording.add_segment(
                        session_id, "ruchi", server_voice.pcm_to_wav(bytes(pcm)), ext="wav"
                    )
                except Exception as exc:
                    log.warning("could not save streamed TTS: %s", exc)


@app.post("/api/voice/stt")
async def voice_stt(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Transcribe Telugu audio.

    Preference order:
      1. Sarvam saaras:v3 (``SARVAM_API_KEY``)
      2. Vosk (offline, no key)
      3. browser SpeechRecognition (frontend fallback)
    Accepts webm/ogg/wav/mp3.

    When ``session_id`` is provided the uploaded mic audio is appended to
    that session's recording (mixed with Ruchi's TTS into one MP3).
    """
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio file")
    fname = file.filename or "audio.webm"
    ctype = file.content_type or "audio/webm"

    recording.add_segment(session_id, "user", audio)

    # 1) Sarvam
    if server_voice.sarvam_configured():
        try:
            transcript = await server_voice.transcribe_sarvam(audio, fname, ctype)
            return {"transcript": transcript, "provider": "sarvam"}
        except Exception as exc:
            log.warning("Sarvam STT failed (%s) — falling back to Vosk", exc)

    # 2) Vosk
    caps = server_voice.capabilities()
    if caps.get("vosk") and caps.get("vosk_model"):
        try:
            transcript = await asyncio.to_thread(server_voice.transcribe_vosk, audio, fname)
            return {"transcript": transcript, "provider": "vosk"}
        except Exception as exc:
            log.warning("Vosk STT failed: %s", exc)

    # 3) Browser
    return JSONResponse(
        status_code=200,
        content={"transcript": "", "fallback": "browser", "reason": "no server STT available"},
    )


# ---------------------------------------------------------------- realtime STT
@app.websocket("/api/voice/stream")
async def voice_stream(ws: WebSocket):
    """Realtime turn detection + transcription.

    The browser sends raw 16 kHz mono linear16 PCM frames; we relay them
    to Sarvam's realtime WebSocket and stream back partial/final
    transcripts and VAD (speech start/end) events. Sarvam's server-side
    VAD is what ends the user's turn, so there's no client silence guess.
    """
    await ws.accept()
    if not stream_stt.configured():
        await ws.send_json({"event": "error", "message": "Sarvam not configured"})
        await ws.close()
        return

    session_id = ws.query_params.get("session_id")
    sarvam = stream_stt.SarvamStream()
    try:
        await sarvam.connect()
    except Exception as exc:  # noqa: BLE001
        log.warning("Sarvam realtime connect failed: %s", exc)
        try:
            await ws.send_json({"event": "error", "message": f"stt connect failed: {exc}"})
        except Exception:  # noqa: BLE001
            pass
        await ws.close()
        return

    pcm = bytearray()

    def flush_user_audio() -> None:
        """Save the PCM captured since the last final as one user segment."""
        if not pcm or not session_id:
            pcm.clear()
            return
        try:
            recording.add_segment(
                session_id, "user", server_voice.pcm_to_wav(bytes(pcm)), ext="wav"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save streamed user audio: %s", exc)
        pcm.clear()

    async def browser_to_sarvam() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    return
                data = msg.get("bytes")
                if data:
                    pcm.extend(data)
                    try:
                        await sarvam.send_audio(data)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Sarvam send failed: %s", exc)
                        return
        except WebSocketDisconnect:
            return

    async def sarvam_to_browser() -> None:
        try:
            async for ev in sarvam.events():
                try:
                    await ws.send_json(ev)
                except Exception:  # noqa: BLE001
                    return
                if ev.get("event") == "transcript.final":
                    flush_user_audio()
        except Exception as exc:  # noqa: BLE001
            log.warning("Sarvam recv failed: %s", exc)

    t_in = asyncio.create_task(browser_to_sarvam())
    t_out = asyncio.create_task(sarvam_to_browser())
    try:
        _done, pending = await asyncio.wait(
            {t_in, t_out}, return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await sarvam.close()
        flush_user_audio()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- demo
if DEMO_DIR.is_dir():
    app.mount("/demo", StaticFiles(directory=str(DEMO_DIR), html=True), name="demo")
else:  # pragma: no cover
    log.warning("demo/ folder not found at %s", DEMO_DIR)


@app.on_event("startup")
async def _startup() -> None:
    # Load the recipe catalog once at boot so the first turn is fast.
    recipe_catalog.get_catalog()
    # Memory layer (Postgres + pgvector). Best-effort: a missing DB or
    # embedding model must never stop the server from booting.
    try:
        from voice_agent_functions import memory
        from voice_agent_functions.memory import manager as memory_manager
        from voice_agent_functions.memory import context as memory_context
        from voice_agent_functions.memory import recipe_index

        if not memory_manager.memory_enabled():
            log.info("memory layer disabled (MEMORY_ENABLED=0 or no DATABASE_URL)")
            return
        memory.bootstrap_schema()
        threading.Thread(target=memory_context.warmup, daemon=True).start()
        recipe_index.sync_async()
        log.info("memory layer enabled (Postgres + pgvector)")
    except Exception as exc:  # noqa: BLE001
        log.warning("memory layer unavailable: %s", exc)


@app.on_event("shutdown")
async def _shutdown() -> None:
    try:
        from voice_agent_functions.memory import database as memory_database

        memory_database.close_pool()
    except Exception:  # noqa: BLE001
        pass
    await server_voice.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
