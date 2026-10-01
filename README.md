# Ruchi — Telugu voice cooking assistant

A real-time, hands-free **Telugu** cooking companion. Tap the orb, speak
naturally; Ruchi guides you step by step, sets timers, and remembers you
across sessions.

## What it does

- Listens to your Telugu speech and transcribes it in real time
- Holds a cooking conversation (intake → steps → timers → fixes)
- Speaks every reply aloud in Telugu with expressive delivery
- Loads recipes from a 200-dish South-Indian dataset (or the web)
- Remembers your preferences, restrictions, and history (Postgres)
- Lets you interrupt mid-sentence — it stops and answers the new question

## Architecture

Four layers. Each layer degrades gracefully to the next one down.
<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/e9eda983-d916-47f7-842f-db0ffc9713e7" />


```

### A single voice turn, end to end

1. You speak → mic PCM streams to `/api/voice/stream`.
2. **Sarvam** transcribes live; its VAD decides when your turn ends →
   `transcript.final`.
3. Browser POSTs the text to `/api/voice-agent/sessions/{id}/turn`.
4. Server fetches your recent turns + relevant long-term **memories**
   (local embedding + pgvector, ~30 ms) and builds the prompt:
   *system prompt + recipe truth + memories*.
5. **Gemini** replies, optionally calling tools (`load_recipe`,
   `set_servings`, `next_step`, `start_timer`, …).
6. The turn + tool results are saved (Postgres + Markdown log); memory
   extraction runs in the background.
7. Browser POSTs the reply to `/api/voice/tts`; **ElevenLabs** streams
   Telugu audio that starts playing in ~1 s.

## Required API keys

| Key(s) | Used for | Where to get it | Without it |
|---|---|---|---|
| `GOOGLE_API_KEY` | Gemini LLM (conversation + tools) and background memory extraction | https://aistudio.google.com | Mock runner (limited canned replies) |
| `ELEVEN_API_KEY` + `ELEVEN_VOICE_ID` | Expressive Telugu TTS | https://elevenlabs.io | edge-tts (free, no key), then browser voice |
| `SARVAM_API_KEY` | Telugu speech-to-text (realtime + batch) | https://sarvam.ai | Vosk offline model, then browser STT |

All keys go in `.env` (copy `.env.example`). Everything else —
Vosk, edge-tts, Postgres, embeddings — runs locally and free.

## Setup from scratch

### 1. Prerequisites

- **Python 3.10+** (`python3 --version`)
- **Docker** (only for the memory database) — `docker --version`
- **Firefox or Chrome**, with a working microphone
- Open `http://127.0.0.1:8000/demo/` (localhost is required for mic access)

### 2. Clone and create the Python environment

```bash
git clone https://github.com/aegion-dynamic/ruchi-build-v1.git
cd ruchi-build-v1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and paste your API keys (see table above)
```

### 4. Download the Vosk Telugu model (one time, ~50 MB)

```bash
mkdir -p models && cd models
curl -L -o vosk-te.zip https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip
unzip vosk-te.zip && rm vosk-te.zip && cd ..
```

Skip this if you only want browser speech recognition (set nothing — the
server falls back automatically).

### 5. Start the memory database (one time)

```bash
docker compose -f infra/docker-compose.yaml up -d postgres
```

The server creates the tables itself on first boot, syncs the 200 recipes,
and downloads the small local embedding model (~220 MB) once. Set
`MEMORY_ENABLED=0` in `.env` to skip all of this.

### 6. Run the server

```bash
./run.sh                       # http://127.0.0.1:8000
PORT=8001 ./run.sh             # custom port
```

Useful URLs: demo → `/demo/`, API docs → `/docs`, health → `/api/health`.

### 7. Use it

1. Open `http://127.0.0.1:8000/demo/` (hard-refresh once: `Ctrl+Shift+R`).
2. Allow the microphone **once** (tick "remember this decision").
3. Tap the **orb** — Ruchi greets you in Telugu.
4. Speak, e.g. *"నాకు ఇడ్లీ చేయాలి"*. Interrupt her anytime; she stops
   and answers the new question.

## Running & testing

```bash
./run.sh                                   # start the server
.venv/bin/python -m pytest tests/ -q      # test suite
curl http://127.0.0.1:8000/api/health     # health check
docker compose -f infra/docker-compose.yaml ps postgres   # DB status
```

## Session recordings → MinIO

Every session's mic + TTS audio is mixed into `recordings/<id>.mp3` and
**mirrored to MinIO** automatically (bucket `recordings`, public-read):

- `GET /api/voice-agent/sessions/{id}/recording` → downloads the MP3 and
  returns its MinIO link in the `X-Minio-URL` header.
- `GET /api/voice-agent/recordings` → lists recordings with `minio_url`.

Config in `.env` (`MINIO_ENABLED=1`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`,
`MINIO_BUCKET`). MinIO console: `http://127.0.0.1:9001`.

## Repository layout

```text
.
├── app.py                     FastAPI app: voice + agent + recipe + memory endpoints
├── run.sh                     start the server (./run.sh, PORT=8001 ./run.sh)
├── requirements.txt           Python deps
├── pytest.ini                 pytest config (testpaths = tests)
├── demo/                      browser client (index.html, app.js, styles.css)
├── voice_agent_functions/     the agent package (see its README.md)
│   ├── routes.py              /api/voice-agent/* (sessions, turns, tools, memory)
│   ├── runner.py              Gemini function-calling loop + latency controls
│   ├── dispatcher.py          tool name → Python handler
│   ├── state.py               VoiceSession / RecipeContext
│   ├── cooking_tools.py       15 cook tools (recipe, steps, timers, …)
│   ├── intake_tools.py        recipe-recording interview tools
│   ├── tool_schemas.py        Gemini functionDeclarations
│   ├── mock_runner.py         offline fallback when no Gemini key
│   ├── server_voice.py        TTS/STT providers + capabilities + TTS formatting
│   ├── stream_stt.py          realtime STT proxy (Sarvam WebSocket → browser WS)
│   ├── recording.py           per-session audio segments → session MP3
│   ├── transcript.py          Markdown conversation logs
│   ├── recipe_catalog.py      200-recipe JSON dataset loader + fuzzy search
│   ├── recipe_engine.py       deterministic recipe truth (quantities, heat…)
│   ├── online_recipe.py       web fallback for dishes outside the dataset
│   ├── prompts/               LLM system prompts (cook / intake)
│   ├── prompts_loader.py      loads prompts/*.txt
│   ├── memory/                Postgres + pgvector persistence/memory/RAG
│   └── README.md              package docs (ported tool specs)
├── tests/                     pytest suite (mirrors the package above)
├── data/                      runtime data (see data/README.md)
│   └── recipes.json           the 200-recipe South-Indian dataset (tracked)
├── models/                    Vosk Telugu STT model (downloaded, git-ignored)
├── infra/                     local infra (Postgres+pgvector, LiveKit stack)
├── docs/                      design notes (ARCHITECTURE.md/.pdf, explained.md)
├── transcripts/               human-readable conversation logs (runtime)
└── recordings/                per-session audio (runtime)
```

## Key runtime behaviors

- **Voice loop**: mic PCM → Sarvam realtime STT (`/api/voice/stream`) →
  Gemini tool-use turn (`/turn`) → streamed ElevenLabs Telugu TTS.
  Everything degrades gracefully (Vosk → browser, edge-tts → browser,
  mock runner without a Gemini key).
- **Ruchi speaks only Telugu**, greets once per session, never stops the
  conversation herself, and supports mid-speech **barge-in**.
- **Recipe truth lives in code** (`recipe_engine` + session state):
  exact quantities, Telugu heat phrases, English utensils, timers,
  completed/parallel steps. The LLM only verbalizes it.
- **Memory** (`MEMORY_ENABLED=1`): sessions and turns persist to local
  Postgres+pgvector; durable facts are extracted in the background and
  retrieved by semantic similarity on every turn. No DB → still works,
  just without memory.


