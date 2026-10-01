# Ruchi — Telugu voice cooking assistant

A real-time, hands-free **Telugu** cooking companion. Speak naturally;
Ruchi guides you step by step, sets timers, and remembers you.

```text
Browser (demo/) ──mic PCM──▶ /api/voice/stream (Sarvam realtime STT)
        ▲──played audio ◀── /api/voice/tts (ElevenLabs streaming)
        │                    /api/voice-agent/... (Gemini function-calling
        │                     tools: recipe, steps, timers, memory tools)
        │                              │
        │                    ┌─────────▼──────────┐
        │                    │ Postgres + pgvector │  sessions, turns,
        │                    │ (local Docker)      │  long-term memories,
        │                    └─────────────────────┘  200-recipe catalog
```

## Quickstart

```bash
./run.sh                       # server on 127.0.0.1:8000 → http://127.0.0.1:8000/demo/
docker compose -f infra/docker-compose.yaml up -d postgres   # memory database
```

Copy `.env.example` to `.env` and fill in API keys. Then hard-refresh the
demo in Firefox (`Ctrl+Shift+R`), tap the orb, and speak.

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
│   ├── cooking_tools.py       13 cook tools (recipe, steps, timers, …)
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
├── infra/                     local infra (Postgres+pgvector LiveKit stack)
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

## Docs

- `docs/ARCHITECTURE.md` (and `.pdf`) — design notes.
- `docs/explained.md` — component explainer.
- `voice_agent_functions/README.md` — the original function-tool spec.
