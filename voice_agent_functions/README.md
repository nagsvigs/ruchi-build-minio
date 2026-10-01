# `voice_agent_functions` — function-calling tools for Ruchi

This folder is the FastAPI-side counterpart to
[`ruchi-voice-agent` PR #1](https://github.com/aegion-dynamic/ruchi-voice-agent/pull/1).

That PR documented a **two-agent, two-tool-set** voice design on the
LiveKit worker:

| Mode | When it runs | Tools the LLM can call |
|---|---|---|
| `intake` | room named `ruchi-intake-...`, scripted recipe-recording flow | `save_answer`, `finish_interview` |
| `cook`   | room named `ruchi-cook-...`,   freeform cooking-companion flow  | `next_step`, `previous_step`, `goto_step`, `start_timer`, `cancel_timer`, `pause`, `resume`, `report_issue`, `resolve_issue`, `finish_cooking` |

This package ports both tool sets to the FastAPI backend so the React
frontend can drive the same flows over plain HTTP without depending on
LiveKit. The PR's docs (`GUIDE.md`, `RECIPE_INTAKE_AGENT.md`,
`COOKING_AGENT.md`, `STEP_NAVIGATION.md`, `TIMER.md`, `PAUSE_RESUME.md`,
`FIX_MY_DISH.md`) are the source of truth for tool behavior — the
Python handlers below implement exactly what those docs say, and the
phrases they return are the canned lines the docs prescribe verbatim.

## Layout

```
voice_agent_functions/                ← project root (this folder)
├── README.md                         ← you are here
├── app.py                            ← standalone FastAPI entry point
├── run.sh                            ← convenience: ./run.sh to start the server
├── .env.example                      ← env template (copy to .env)
├── demo/                             ← interactive HTML/JS demo
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── voice_agent_functions/            ← the package
    ├── __init__.py
    ├── state.py              # VoiceSession, RecipeContext, INTAKE_FIELDS
    ├── intake_tools.py       # save_answer, finish_interview
    ├── cooking_tools.py      # 10 cooking-companion tools
    ├── tool_schemas.py       # Gemini functionDeclarations
    ├── dispatcher.py         # routes functionCall → handler
    ├── prompts_loader.py
    ├── prompts/
    │   ├── cooking.txt               ← from PR #1 commit 8414c1b
    │   ├── intake.txt                ← from RECIPE_INTAKE_AGENT.md
    │   └── cooking_companion.txt     ← from COOKING_AGENT.md
    ├── runner.py             # Gemini tool-use loop (production path)
    ├── mock_runner.py        # rule-based runner (dev, no API key)
    ├── routes.py             # FastAPI router (/api/voice-agent/...)
    └── tests/                # 47 unit tests
```

## How to start using it (3 ways)

> **Pick whichever fits.** All three work **without any API keys** —
> the mock runner handles the 12 tools, `edge-tts` gives free Telugu
> voice output, and Vosk gives offline Telugu voice input.

### ① Quickest — the live Telugu cooking companion demo

**Important:** open the demo at **http://127.0.0.1:8000/demo/**, *not*
as a `file://` path. Browsers only allow microphone access on
`localhost` or HTTPS — `file://` blocks the mic entirely.

```bash
cd ~/Projects/voice_agent_functions

# 1) Start the server (one terminal):
./run.sh
# → serves http://127.0.0.1:8000

# 2) Open the demo — use the localhost URL:
xdg-open http://127.0.0.1:8000/demo/
```

You get a **working voice assistant**:

| Feature | How it works |
|---|---|
| 🎤 **Voice input** | Click the orb or mic → speak Telugu → auto-stops on silence → Vosk transcribes offline → fires the matching tool |
| 🔊 **Voice output** | Every reply is spoken with `edge-tts` Telugu neural voice (`te-IN-ShrutiNeural`) |
| 💬 **Live transcript** | Your words + Ruchi's replies appear as chat bubbles, with an amber pill showing each tool that fired |
| 📖 **Recipe panel** | Left rail shows all steps with the current one highlighted (updates on `next_step`/`previous_step`/`goto_step`) |
| 📝 **Intake panel** | In intake mode, all 8 fields shown as a checklist |
| ⚡ **All 12 tools** | Right panel has a chip per tool; click to fire it directly |
| 📜 **Function log** | Full args + result for every tool call, color-coded |

The demo also accepts `?api=` to point at a different host:

```
http://127.0.0.1:8000/demo/?api=http://192.168.1.5:8000
```

#### Voice setup (one time)

The server needs two Python packages and one model file:

```bash
cd ~/Projects/voice_agent_functions
.venv/bin/pip install edge-tts vosk

# Vosk Telugu model (~60 MB download, 122 MB unpacked):
mkdir -p models && cd models
curl -L -o vosk-te.zip https://alphacephei.com/vosk/models/vosk-model-small-te-0.42.zip
unzip vosk-te.zip && rm vosk-te.zip
cd ..
```

Also needs `ffmpeg` (already present on most systems) to convert the
browser's webm recording to WAV for Vosk:

```bash
which ffmpeg || sudo pacman -S ffmpeg
```

Check everything with:

```bash
curl http://127.0.0.1:8000/api/voice/capabilities
# {"edge_tts":true,"vosk":true,"vosk_model":true,"ffmpeg":true,...}
```

If any of these is missing, the demo falls back to browser
`speechSynthesis` / `SpeechRecognition` and shows a diagnostic bubble
at startup telling you exactly what's active.

### ② React frontend (full UI with chat + tools + LiveKit)

If you have `ruchi-basicfullstack` cloned:

```bash
# Terminal 1 — backend (uses the full app.py with chat + voice + tools)
cd /path/to/ruchi-basicfullstack/backend
PYTHONPATH=. .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd /path/to/ruchi-basicfullstack/frontend
npm install        # one time
npm run dev        # → http://localhost:5173
```

The UI has three tabs:
- **💬 Chat** — text + browser STT/TTS, works with zero keys.
- **🛠 Tools** — voice orb, live function log, quick-reply chips.
- **📞 LiveKit** — only enabled when `LIVEKIT_*` keys are set; otherwise
  shows a clean setup guide.

### ③ CLI one-shot — call one tool directly

```bash
cd ~/Projects/voice_agent_functions
.venv/bin/python -c "
from voice_agent_functions.dispatcher import dispatch
from voice_agent_functions.state import VoiceSession, RecipeContext
s = VoiceSession(session_id='t', mode='cook',
    recipe=RecipeContext(recipe_name='Paneer Biryani', steps=[
        'Soak the rice for 30 minutes.',
        'Marinate the paneer in yogurt and spices.',
        'Layer rice and paneer in a heavy pot.',
        'Cook on low heat for 20 minutes.',
    ]))
print(dispatch(s, 'next_step', {}))
"
```

Output:

```
{'result': {'type': 'step_changed', 'stepIndex': 1,
            'lead_in': 'Okay, next step.',
            'phrase': 'Okay, next step. Marinate the paneer in yogurt and spices.'},
 'name': 'next_step'}
```

Or run the full test suite:

```bash
.venv/bin/python -m pytest voice_agent_functions/tests -v
# 47 passed in 0.1s
```

## Setup from scratch

```bash
cd ~/Projects/voice_agent_functions

# 1) Create a venv (Arch blocks system-wide pip)
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install fastapi httpx python-dotenv pytest pytest-asyncio uvicorn

# 2) Run the tests
.venv/bin/python -m pytest voice_agent_functions/tests -v

# 3) Start the server
./run.sh                          # → http://127.0.0.1:8000/api/health

# 4) Open the demo
xdg-open demo/index.html
```

## The 12 tools

### Intake mode (`mode: "intake"`)

| Tool | Args | Behavior |
|---|---|---|
| `save_answer` | `field`, `value` | Stores the answer, advances the field pointer, publishes `field_saved`. On `steps` it also fires `finish_interview`. |
| `finish_interview` | — | Publishes `interview_complete` with every collected answer, freezes the session. |

Fields walked in order: `name` → `time` → `servings` → `level` →
`ingredients` → `utensils` → `tips` (optional — empty string is valid)
→ `steps`.

### Cooking-companion mode (`mode: "cook"`)

| Tool | Args | Behavior |
|---|---|---|
| `next_step` | — | Increments `stepIndex`, publishes `step_changed`. |
| `previous_step` | — | Decrements `stepIndex`, publishes `step_changed`. |
| `goto_step` | `step` (1-indexed) | Jumps to the named step. |
| `start_timer` | `seconds`, `label` | Background asyncio task, publishes `timer_started`; on expiry publishes `timer_expired`. |
| `cancel_timer` | — | Cancels active timer, publishes `timer_cancelled`. |
| `pause` | — | Sets `paused = True`, publishes `paused`. |
| `resume` | — | Clears `paused`, publishes `resumed` with current step text. |
| `report_issue` | `issue`, `details` | Stores issue, publishes `issue_reported`. Categories: `spicy`, `salty`, `watery`, `burnt`, `bland`, `other`. |
| `resolve_issue` | — | Clears active issue, publishes `issue_resolved`. |
| `finish_cooking` | — | Cancels timer, publishes `cooking_finished`. |

Every handler:
- **Validates** args → returns `{error}` envelope on failure
- **Mutates** the session state
- **Appends** to `session.events` (stable list of `{"type": ...}` dicts
  the connected frontend listens for)
- **Returns** a `phrase` field with the canned line the PR docs
  prescribe verbatim

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/voice-agent/sessions` | Create `{mode, recipe?}`. |
| `GET` | `/api/voice-agent/sessions/{sid}` | Current state + event log. |
| `DELETE` | `/api/voice-agent/sessions/{sid}` | Tear down (cancels timers). |
| `PUT` | `/api/voice-agent/sessions/{sid}/recipe` | Swap the recipe mid-session. |
| `POST` | `/api/voice-agent/sessions/{sid}/turn` | Run one user utterance (Gemini or mock). |
| `POST` | `/api/voice-agent/sessions/{sid}/call` | Fire a tool **directly**, bypassing the LLM. |
| `POST` | `/api/voice-agent/sessions/{sid}/save_answer` | Frontend shortcut — bypass LLM. |
| `GET` | `/api/voice-agent/tools?mode=intake\|cook` | The Gemini function-declaration schemas. |
| `GET` | `/api/voice-agent/prompts` | Raw system-prompt text for each mode. |
| `GET` | `/api/voice-agent/transcript` | Full conversation transcript (Markdown). |
| `GET` | `/api/voice-agent/transcript/files` | List transcript files on disk. |
| `DELETE` | `/api/voice-agent/transcript` | Clear all transcripts. |

When `GOOGLE_API_KEY` is unset, `/turn` falls back to `mock_runner.py`
so the demo works with zero configuration.

## Conversation transcript

Every exchange between you and Ruchi is appended to
**`transcripts/conversation.md`** (and a per-day `YYYY-MM-DD.md`) in
readable Markdown:

```markdown
### 2026-09-18 00:05:01 · session `d31aa222` · cook

**You:** మొదటి స్టెప్ పూర్తయింది, తర్వాత ఏంటి?

**Ruchi:** ఇప్పుడు దోసెల పెనం బాగా వేడెక్కనివ్వండి...

`tools: next_step()`

_provider: gemini-3.5-flash-lite_
```

The demo has a **📝 Transcript** tab that shows it live with Refresh /
Clear buttons. Config:

| Env var | Default | Purpose |
|---|---|---|
| `TRANSCRIPT_ENABLED` | `1` | set `0` to disable |
| `TRANSCRIPT_DIR` | `./transcripts` | output directory |

```bash
curl http://127.0.0.1:8000/api/voice-agent/transcript          # read it
curl http://127.0.0.1:8000/api/voice-agent/transcript/files    # list files
curl -X DELETE http://127.0.0.1:8000/api/voice-agent/transcript # clear
```

## Environment variables

All optional — the demo works with zero config. Copy `.env.example`
to `.env` and fill in what you have:

```bash
cp .env.example .env
```

| Variable | Purpose | Without it |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini function-calling | keyword mock runner |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | same default |
| `SARVAM_API_KEY` | Sarvam Telugu STT | Vosk (offline) |
| `SARVAM_MODEL` | `saaras:v3` | same default |
| `ELEVEN_API_KEY` + `ELEVEN_VOICE_ID` | ElevenLabs TTS | edge-tts (free) |
| `ELEVEN_MODEL_ID` | `eleven_v3_conversational` | same default |
| `LIVEKIT_URL` + `LIVEKIT_API_KEY` + `LIVEKIT_API_SECRET` | LiveKit realtime worker | disabled |

### Provider fallback chains

Each capability tries the best option first, then falls back:

```
Voice output:  ElevenLabs (eleven_v3_conversational)
                 → edge-tts (te-IN-ShrutiNeural)
                   → browser speechSynthesis

Voice input:   Sarvam (saaras:v3, te-IN)
                 → Vosk (offline Telugu model)
                   → browser SpeechRecognition

LLM:           Gemini (function-calling, 12 tools)
                 → keyword mock runner
```

Check which are active at any time:

```bash
curl http://127.0.0.1:8000/api/voice/capabilities
```

The demo shows the active providers in a system bubble on load.

> **Note:** older Gemini models like `gemini-2.5-flash-lite` and
> `gemini-2.0-flash` have been retired and return 404. Use
> `gemini-3.5-flash-lite` (the current default). The runner also
> auto-falls-back through `gemini-3.5-flash` → `gemini-3.1-flash-lite`
> → `gemini-flash-latest`.

## Design rules (from the PR docs)

1. **The model decides when to call each tool; the handler just runs
   it.** `dispatcher.dispatch` does no intent classification.
2. **Templated-response-first after every tool call.** Each handler
   returns a `phrase` field with the canned line from the docs —
   no extra LLM round-trip to confirm.
3. **Issue categories are a fixed enum of 6.** `report_issue` raises
   on anything else; the dispatcher turns the error into a Gemini
   `functionResponse.error`.
4. **Out-of-range step navigation is a no-op, not an error.** Going
   past the last step returns
   `{"type": "no_op", "phrase": "That's the last step — want me to wrap up?"}`.
5. **Pause is the trickiest judgment call.** The Python handler is
   deliberately dumb — the schema description explicitly tells the
   model "DO NOT call this for 'should I pause the stirring?'".
6. **`tips` is the only optional intake field.** Skip phrases are
   normalized to `""` so `interview_complete.answers` is always a
   stable shape.

## References

- [`ruchi-voice-agent` PR #1](https://github.com/aegion-dynamic/ruchi-voice-agent/pull/1)
- [Gemini function-calling docs](https://ai.google.dev/gemini-api/docs/function-calling)
