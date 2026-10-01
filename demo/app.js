/* ============================================================
   Ruchi — Telugu Cooking Companion
   Live voice assistant + all 12 function-calling tools
   ============================================================ */

const API_BASE =
  (location.search.match(/api=([^&]+)/) || [,
    location.protocol.startsWith('http') ? location.origin : 'http://localhost:8000'
  ])[1];

// Recipes are no longer hardcoded here. Ruchi loads them from the
// server-side catalog by voice (load_recipe tool). RECIPE is whatever
// recipe the session currently has loaded, or null before that.
let RECIPE = null;   // current recipe (loaded from the catalog)

const INTAKE_FIELDS = [
  { name: 'name',        question: "What's the name of your recipe?" },
  { name: 'time',        question: 'About how long does it take to cook?' },
  { name: 'servings',    question: 'How many people does it serve?' },
  { name: 'level',       question: 'How difficult is it — easy, medium, or hard?' },
  { name: 'ingredients', question: 'List the ingredients, with approximate quantities.' },
  { name: 'utensils',    question: 'What utensils do you need?' },
  { name: 'tips',        question: 'Any quick tips you want to add? (optional)' },
  { name: 'steps',       question: 'Now walk me through all the steps.' },
];

const state = {
  sessionId: null,
  mode: 'cook',
  pollHandle: null,
  busy: false,
  listening: false,
  recognition: null,
  voices: [],
  caps: null,
  // MediaRecorder path
  recorder: null,
  recordStream: null,
  micStream: null,     // acquired once, reused for every turn
  // Realtime streaming STT (Vosk) + LiveKit semantic turn detection
  streaming: false,
  streamWs: null,
  streamCtx: null,
  streamNode: null,
  streamSource: null,
  streamRetries: 0,
  speaking: false,     // true while Ruchi's TTS is playing
  bargeIn: false,      // true when the user interrupted Ruchi
  turnAbort: null,     // AbortController for the in-flight turn
  pendingBargeText: null,  // new query to run after interrupting Ruchi
  bargeWatchdog: null, // safety timer for a stuck turn
  aiText: '',          // the assistant text currently being spoken
  recordChunks: [],
  recordStart: 0,
  silenceTimer: null,
  analyser: null,
  audioCtx: null,
  orbState: 'idle',
  // Conversation mode (hands-free loop)
  conversationMode: false,
  greeted: false,      // Ruchi speaks her greeting only once, on first orb tap
  paused: false,       // set by the `pause` tool / "stop" intent
  emptyTurns: 0,       // consecutive turns with no speech
  finishRequested: false,  // set by the finish_cooking tool
  // Event-driven state
  lastState: null,     // most recent session snapshot
  eventCount: 0,       // how many events we've already applied
  // Live timer
  timerHandle: null,
  timerEndsAt: null,
  timerLabel: null,
  timerFired: false,   // guards against double-firing (client + server)
};

// ---------- DOM helpers --------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- Server health ------------------------------------------------
async function ping() {
  try {
    const r = await fetch(`${API_BASE}/api/health`);
    if (!r.ok) throw new Error('not ok');
    const j = await r.json();
    const pill = $('#server-pill');
    pill.textContent = 'live';
    pill.classList.remove('dead');
    pill.classList.add('live');
  } catch {
    const pill = $('#server-pill');
    pill.textContent = 'offline';
    pill.classList.remove('live');
    pill.classList.add('dead');
  }
}

// ---------- Server voice capabilities ---------------------------------
async function loadCaps() {
  try {
    const r = await fetch(`${API_BASE}/api/voice/capabilities`);
    state.caps = await r.json();
  } catch {
    state.caps = { edge_tts: false, vosk: false, vosk_model: false, ffmpeg: false };
  }
  return state.caps;
}

// ---------- Speech synthesis (edge-tts server → browser fallback) -----
function loadVoices() {
  return new Promise((resolve) => {
    let v = window.speechSynthesis?.getVoices() || [];
    if (v.length) return resolve(v);
    let tries = 0;
    const id = setInterval(() => {
      v = window.speechSynthesis?.getVoices() || [];
      if (v.length || ++tries > 10) { clearInterval(id); resolve(v); }
    }, 200);
  });
}

function pickTeluguVoice() {
  const voices = state.voices || [];
  return (
    voices.find((v) => /te-IN|telugu/i.test(v.lang + v.name)) ||
    voices.find((v) => /hi-IN|hindi|india/i.test(v.lang + v.name)) ||
    voices.find((v) => v.lang?.toLowerCase().startsWith('en')) ||
    null
  );
}

function speakBrowser(text) {
  return new Promise((resolve) => {
    if (!text || !window.speechSynthesis) return resolve();
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(stripTags(text));
    const v = pickTeluguVoice();
    if (v) u.voice = v;
    u.lang = v?.lang || 'te-IN';
    u.rate = 0.95;
    u.onstart = () => setOrb('speaking', 'speaking…');
    u.onend = () => { resolve(); };
    u.onerror = () => { resolve(); };
    window.speechSynthesis.speak(u);
  });
}

let _ttsAudio = null;      // <audio> element (non-streamed playback)
let _ttsCtx = null;        // AudioContext reused for PCM streaming
let _ttsAbort = null;      // AbortController for the in-flight TTS stream
let _ttsNodes = [];        // scheduled AudioBufferSourceNodes
let _ttsGen = 0;           // bumped on every interrupt (barge-in / new reply)
let _ttsAudioResolve = null; // resolves the <audio> playback promise on stop

function _stopTts() {
  _ttsGen++;  // makes an in-flight playPcmStream return early
  try { if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; } } catch {}
  // Resolve the pending <audio> promise so speak() never hangs after a pause.
  try { if (_ttsAudioResolve) { const r = _ttsAudioResolve; _ttsAudioResolve = null; r(); } } catch {}
  try { if (_ttsAbort) { _ttsAbort.abort(); _ttsAbort = null; } } catch {}
  for (const n of _ttsNodes) { try { n.stop(); } catch {} }
  _ttsNodes = [];
}

function _pcmStreamSupported() {
  return typeof ReadableStream !== 'undefined' &&
    !!(window.AudioContext || window.webkitAudioContext);
}

/** Play a streamed 16 kHz mono PCM response via the Web Audio API,
 *  starting as soon as the first chunk arrives. Returns false if no
 *  audio was produced (so the caller can fall back). */
async function playPcmStream(resp, sampleRate = 16000) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return false;
  if (!_ttsCtx || _ttsCtx.sampleRate !== sampleRate) {
    try { _ttsCtx = new Ctx({ sampleRate }); }
    catch { _ttsCtx = new Ctx(); }
  }
  const ctx = _ttsCtx;
  if (ctx.state === 'suspended') { try { await ctx.resume(); } catch {} }

  const gen = _ttsGen;
  const reader = resp.body.getReader();
  let leftover = new Uint8Array(0);
  let nextStart = ctx.currentTime;
  let got = 0;
  while (true) {
    if (_ttsGen !== gen) return true;  // interrupted by barge-in
    const { done, value } = await reader.read();
    if (done) break;
    let bytes = value;
    if (leftover.length) {
      const merged = new Uint8Array(leftover.length + bytes.length);
      merged.set(leftover, 0); merged.set(bytes, leftover.length);
      bytes = merged; leftover = new Uint8Array(0);
    }
    if (bytes.length % 2) { leftover = bytes.slice(bytes.length - 1); bytes = bytes.slice(0, -1); }
    if (!bytes.length) continue;
    // .slice() gives a fresh buffer at offset 0 so Int16Array is aligned.
    const i16 = new Int16Array(bytes.slice().buffer);
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const buf = ctx.createBuffer(1, f32.length, sampleRate);
    buf.getChannelData(0).set(f32);
    const src = ctx.createBufferSource();
    src.buffer = buf; src.connect(ctx.destination);
    const startAt = Math.max(ctx.currentTime + 0.03, nextStart);
    src.start(startAt);
    nextStart = startAt + buf.duration;
    _ttsNodes.push(src);
    got += i16.length;
  }
  if (!got) return false;
  // Wait until the scheduled audio has played, but bail out early on barge-in.
  // A wall-clock cap prevents an infinite wait if the AudioContext is
  // suspended (autoplay policy) and its clock never advances.
  const deadline = performance.now() + (nextStart - ctx.currentTime) * 1000 + 3000;
  while (nextStart - ctx.currentTime > 0.03) {
    if (_ttsGen !== gen) return true;
    if (performance.now() > deadline) break;
    await wait(40);
  }
  return true;
}

/** Speak `text`. Resolves when playback finishes (or immediately on failure). */
async function speak(text, { interrupt = true } = {}) {
  if (!text) return;
  setOrb('speaking', 'speaking…');
  if (interrupt) _stopTts();
  // Keep the mic live while Ruchi speaks so the user can barge in (echo
  // cancellation suppresses her own voice; the echo filter catches leaks).
  state.speaking = true;
  state.aiText = text;
  try {
    // 1) Prefer server-side TTS (ElevenLabs → edge-tts, chosen server-side)
    if (state.caps?.elevenlabs || state.caps?.edge_tts) {
      // 1a) Streaming PCM (ElevenLabs only) — lowest latency.
      if (state.caps?.elevenlabs && _pcmStreamSupported()) {
        try {
          const ac = new AbortController();
          _ttsAbort = ac;
          const r = await fetch(`${API_BASE}/api/voice/tts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, session_id: state.sessionId, stream: true }),
            signal: ac.signal,
          });
          const ct = r.headers.get('content-type') || '';
          if (ct.includes('audio/pcm')) {
            const rate = parseInt(r.headers.get('X-Sample-Rate') || '16000', 10) || 16000;
          const played = await playPcmStream(r, rate);
          _ttsAbort = null;
          if (played) return;
        }
        _ttsAbort = null;
      } catch (e) {
        _ttsAbort = null;
        if (e.name === 'AbortError') return;  // barge-in/interrupt — stop here
        log('error', 'streaming TTS failed', String(e).slice(0, 120));
      }
    }

    // 1b) Non-streaming server TTS (whole file, then play).
    try {
      const r = await fetch(`${API_BASE}/api/voice/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, session_id: state.sessionId }),
      });
      const ct = r.headers.get('content-type') || '';
      if (ct.includes('audio')) {
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        _ttsAudio = audio;
        await new Promise((resolve) => {
          const done = () => {
            _ttsAudioResolve = null;
            try { URL.revokeObjectURL(url); } catch {}
            resolve();
          };
          _ttsAudioResolve = done;   // lets _stopTts resolve on barge-in
          audio.onended = done;
          audio.onerror = done;
          audio.play().catch(done);
        });
        return;
      }
      // JSON fallback response → use browser
    } catch (e) {
      log('error', 'server TTS failed', String(e).slice(0, 120));
    }
    }

    // 2) Browser fallback
    await speakBrowser(text);
  } finally {
    state.speaking = false;
    state.aiText = '';
  }
}

// ---------- Speech recognition ----------------------------------------
// Primary: MediaRecorder → POST /api/voice/stt (Vosk Telugu, offline).
// This works in every browser over localhost, unlike webkitSpeechRecognition
// which Brave disables and which needs a secure context anyway.
// Secondary: browser SpeechRecognition if server STT is unavailable.

async function startListening() {
  if (state.listening) return;
  // Preferred: realtime streaming STT with server-side turn detection.
  const serverStt = state.caps?.sarvam;
  log('system', 'STT', `sarvam=${!!state.caps?.sarvam} → ${serverStt ? 'streaming' : 'fallback'}`);
  if (serverStt && window.WebSocket && (window.AudioContext || window.webkitAudioContext)) {
    return startStreaming();
  }
  if (state.caps?.vosk && state.caps?.vosk_model && state.caps?.ffmpeg) {
    return startServerListening();
  }
  return startBrowserListening();
}

// ---- server path (MediaRecorder + Vosk) ----
// Acquire the microphone ONCE and reuse the same live stream for every
// turn. Calling getUserMedia() again on each recording is what makes
// some browsers re-prompt for permission; one persistent stream avoids it.
async function getMicStream() {
  const existing = state.micStream;
  if (existing && existing.getAudioTracks().some((t) => t.readyState === 'live')) {
    return existing;
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  state.micStream = stream;
  // If the browser/OS ends the track (device change, revoke), drop the
  // cache so the next turn re-acquires cleanly.
  stream.getAudioTracks().forEach((t) => {
    t.addEventListener('ended', () => {
      if (state.micStream === stream) state.micStream = null;
    });
  });
  return stream;
}

// ---- realtime streaming path (Sarvam WebSocket, server-side turn detection) ----
function _wsUrl() {
  const base = API_BASE.replace(/^http/, 'ws');
  const q = state.sessionId ? `?session_id=${encodeURIComponent(state.sessionId)}` : '';
  return `${base}/api/voice/stream${q}`;
}

async function startStreaming() {
  if (state.streaming) return;
  let mic;
  try {
    mic = await getMicStream();
  } catch (e) {
    log('error', 'microphone blocked', 'Allow mic access for this site, then retry.');
    pushBubble('system', '⚠ Microphone blocked. Click the mic icon in the address bar and allow access, then try again.');
    endConversation();
    return;
  }

  let ws;
  try { ws = new WebSocket(_wsUrl()); }
  catch (e) { return startServerListening(); }

  state.streamWs = ws;
  state.streaming = true;
  state.listening = true;
  _setMicEnabled(true);   // ensure the mic is live at the start
  setOrb('listening', 'వింటున్నాను… (realtime)');
  updateMicUi();
  updateVoicePill();

  ws.binaryType = 'arraybuffer';
  ws.onopen = () => {
    state.streamRetries = 0;
    log('system', 'STT', 'realtime stream open');
    try {
      _startPcmCapture(mic);
      const n = mic.getAudioTracks().length;
      log('system', 'STT', `pcm capture started (${n} track${n === 1 ? '' : 's'}, ctx=${state.streamCtx?.state})`);
    } catch (e) { log('error', 'pcm capture failed', String(e)); }
  };
  ws.onmessage = (ev) => _onStreamEvent(ev);
  ws.onerror = () => log('error', 'stt stream error', 'reconnecting if it closes');
  ws.onclose = () => {
    const wasStreaming = state.streaming;
    stopStreaming();
    if (!(wasStreaming && state.conversationMode && !state.paused)) return;
    // Reconnect the realtime stream rather than falling back to the
    // ungated MediaRecorder path (which would record Ruchi's own voice).
    state.streamRetries = (state.streamRetries || 0) + 1;
    if (state.streamRetries <= 6) {
      pushBubble('system', `Realtime STT dropped — reconnecting (${state.streamRetries})…`);
      setTimeout(() => {
        if (state.conversationMode && !state.paused) startStreaming();
      }, 600);
    } else {
      pushBubble('system', 'Realtime STT unavailable — using fallback.');
      startServerListening();
    }
  };
}

function _setMicEnabled(on) {
  try {
    (state.micStream?.getAudioTracks() || []).forEach((t) => { t.enabled = on; });
  } catch {}
}

/** Create/resume the AudioContexts during a user gesture.
 *  Browsers start AudioContexts suspended when created outside a gesture,
 *  which silently stops the mic capture. Call this from click handlers. */
function _unlockAudio() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) return;
  try {
    if (!state.streamCtx || state.streamCtx.state === 'closed') {
      state.streamCtx = new Ctx({ sampleRate: 16000 });
    }
    if (state.streamCtx.state === 'suspended') state.streamCtx.resume();
  } catch {}
  try {
    if (!_ttsCtx || _ttsCtx.state === 'closed') _ttsCtx = new Ctx();
    if (_ttsCtx.state === 'suspended') _ttsCtx.resume();
  } catch {}
}

function _startPcmCapture(mic) {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  // Reuse one AudioContext across sessions — creating/closing one per orb
  // press eventually hits the browser's AudioContext limit.
  if (!state.streamCtx || state.streamCtx.state === 'closed') {
    state.streamCtx = new Ctx({ sampleRate: 16000 });
  }
  const ctx = state.streamCtx;
  if (ctx.state === 'suspended') { try { ctx.resume(); } catch {} }
  // Drop any previous capture nodes (e.g. after a reconnect).
  try { if (state.streamNode) state.streamNode.disconnect(); } catch {}
  try { if (state.streamSource) state.streamSource.disconnect(); } catch {}
  const source = ctx.createMediaStreamSource(mic);
  const node = ctx.createScriptProcessor(2048, 1, 1);
  const inRate = ctx.sampleRate || 16000;
  const ratio = inRate / 16000;
  log('system', 'STT', `audio ctx rate=${inRate}Hz${inRate !== 16000 ? ' (resampling to 16k)' : ''}`);
  node.onaudioprocess = (e) => {
    const ws = state.streamWs;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Keep the mic live even while Ruchi speaks, so the user can barge in
    // (echo cancellation suppresses Ruchi's own voice).
    const f32 = e.inputBuffer.getChannelData(0);
    // Sarvam needs 16 kHz mono. Browsers may ignore the AudioContext
    // sampleRate hint, so resample linearly when needed.
    let samples = f32;
    if (ratio !== 1) {
      const outLen = Math.floor(f32.length / ratio);
      samples = new Float32Array(outLen);
      for (let i = 0; i < outLen; i++) {
        const idx = i * ratio;
        const i0 = Math.floor(idx);
        const i1 = Math.min(i0 + 1, f32.length - 1);
        const frac = idx - i0;
        samples[i] = f32[i0] * (1 - frac) + f32[i1] * frac;
      }
    }
    const i16 = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(i16.buffer);
  };
  // ScriptProcessor only runs when connected to the destination; route it
  // through a zero-gain node so the mic isn't played back (no feedback).
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(node);
  node.connect(mute);
  mute.connect(ctx.destination);
  state.streamSource = source;
  state.streamNode = node;
}

/** True when `partial` is mostly the words Ruchi is currently speaking
 *  (i.e. speaker echo picked up by the mic), not a new user query. */
function _looksLikeEcho(partial, aiText) {
  const a = (aiText || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!a.length) return false;
  const set = new Set(a);
  const p = (partial || '').toLowerCase().split(/\s+/).filter(Boolean);
  if (!p.length) return false;
  const common = p.filter((w) => set.has(w)).length;
  return common / p.length > 0.6;
}

function _onStreamEvent(ev) {
  let msg;
  try { msg = JSON.parse(ev.data); } catch { return; }
  const type = msg.event || msg.type;
  if (type === 'vad.speech_start') {
    setOrb('listening', 'వింటున్నాను…');
  } else if (type === 'transcript.partial') {
    const t = (msg.text || '').trim();
    if (t) {
      setOrb('listening', t.slice(-48));
      // Barge-in: the user is talking while Ruchi is speaking — stop her
      // immediately and cancel the in-flight turn. Ignore text that matches
      // Ruchi's own words (speaker echo) so she doesn't interrupt herself.
      if (state.speaking && !_looksLikeEcho(t, state.aiText)) {
        _stopTts();
        state.bargeIn = true;
        if (state.turnAbort) { try { state.turnAbort.abort(); } catch {} }
      }
    }
  } else if (type === 'transcript.final') {
    const text = (msg.text || '').trim();
    if (!text || state.paused || state.finishRequested) return;
    log('system', 'STT', `final (busy=${state.busy} speaking=${state.speaking}): ${text.slice(0, 40)}`);
    // If the busy flag is stale (no in-flight turn, nothing speaking),
    // clear it so the new query runs immediately.
    if (state.busy && !state.speaking && !state.turnAbort) {
      log('error', 'turn', 'clearing stale busy flag');
      state.busy = false;
    }
    if (state.busy) {
      // User interrupted Ruchi's answer — drop the old turn and run the
      // new one as soon as the current turn unwinds.
      state.pendingBargeText = text;
      _stopTts();
      if (state.turnAbort) { try { state.turnAbort.abort(); } catch {} }
      // Safety net: if the in-flight turn is wedged, force it to unwind.
      clearTimeout(state.bargeWatchdog);
      state.bargeWatchdog = setTimeout(() => {
        if (state.busy && state.pendingBargeText) {
          log('error', 'turn', 'stuck turn — forcing reset');
          state.busy = false;
          state.speaking = false;
          const t = state.pendingBargeText;
          state.pendingBargeText = null;
          sendMessage(t);
        }
      }, 6000);
      return;
    }
    log('success', 'STT (realtime)', { transcript: text });
    $('#input-text').value = text;
    sendMessage(text);
  } else if (type === 'error') {
    log('error', 'stt stream', msg.message || 'error');
  }
}

function stopStreaming() {
  state.streaming = false;
  state.listening = false;
  try { if (state.streamNode) state.streamNode.disconnect(); } catch {}
  try { if (state.streamSource) state.streamSource.disconnect(); } catch {}
  // Keep state.streamCtx alive for the next session (reused, not closed).
  state.streamNode = null; state.streamSource = null;
  const ws = state.streamWs;
  state.streamWs = null;
  if (ws) { try { ws.onclose = null; ws.onmessage = null; ws.close(); } catch {} }
}

async function startServerListening() {
  let stream;
  try {
    stream = await getMicStream();
  } catch (e) {
    log('error', 'microphone blocked', 'Allow mic access for this site, then retry.');
    pushBubble('system', '⚠ Microphone blocked. Click the mic icon in the address bar and allow access, then try again.');
    endConversation();
    return;
  }
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus'
    : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  state.listening = true;
  state.recorder = rec;
  state.recordStream = stream;
  state.recordChunks = chunks;

  // End-of-turn detection: adaptive noise floor + shorter trailing
  // silence (700ms, was 1200ms) so Ruchi replies faster, plus a minimum
  // speech duration so a cough/click doesn't end the turn.
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const src = ac.createMediaStreamSource(stream);
  const analyser = ac.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  const data = new Uint8Array(analyser.fftSize);
  state.analyser = analyser;
  state.audioCtx = ac;

  const TRAILING_SILENCE_MS = 700;
  const MIN_SPEECH_MS = 250;
  const MIN_THRESHOLD = 6;
  let spoke = false;
  let speechStart = 0;
  let lastLoud = performance.now();
  let noiseFloor = 2;  // running estimate of ambient peak
  const clearSilenceTimer = () => {
    if (state.silenceTimer) { clearInterval(state.silenceTimer); state.silenceTimer = null; }
  };
  const tick = () => {
    if (!state.listening) { clearSilenceTimer(); return; }
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i] - 128));
    const now = performance.now();
    const threshold = Math.max(MIN_THRESHOLD, noiseFloor * 2.5);
    const loud = peak > threshold;
    if (loud) {
      if (!spoke) { state.emptyTurns = 0; speechStart = now; }  // speech began
      spoke = true;
      lastLoud = now;
    } else if (!spoke) {
      // Adapt the noise floor only while the user isn't speaking.
      noiseFloor = noiseFloor * 0.95 + peak * 0.05;
    }
    if (spoke && now - lastLoud > TRAILING_SILENCE_MS && now - speechStart > MIN_SPEECH_MS) {
      stopListening(); return;
    }
    // No speech at all: give more time in conversation mode.
    const idleLimit = state.conversationMode ? 10000 : 8000;
    if (!spoke && now - state.recordStart > idleLimit) { stopListening(); return; }
  };

  rec.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
  rec.onstop = () => transcribeAndSend();
  rec.start();
  state.recordStart = performance.now();
  setOrb('listening', state.conversationMode ? 'వింటున్నాను… (hands-free)' : 'వింటున్నాను… (speak, auto-stops)');
  updateMicUi();
  updateVoicePill();
  // Use a timer (not requestAnimationFrame) so silence detection keeps
  // running even when the tab is backgrounded / throttled.
  clearSilenceTimer();
  state.silenceTimer = setInterval(tick, 100);
}

async function transcribeAndSend() {
  const chunks = state.recordChunks || [];
  const blob = new Blob(chunks, { type: 'audio/webm' });
  // cleanup — close the analyser but KEEP the mic stream alive so the
  // browser doesn't re-prompt for permission on the next turn.
  if (state.silenceTimer) { clearInterval(state.silenceTimer); state.silenceTimer = null; }
  try { state.audioCtx?.close(); } catch {}
  state.recorder = null; state.recordStream = null; state.audioCtx = null;

  if (!blob.size) { afterTurn(); return; }

  setOrb('thinking', 'transcribing…');
  try {
    const fd = new FormData();
    fd.append('file', blob, 'voice.webm');
    if (state.sessionId) fd.append('session_id', state.sessionId);
    const r = await fetch(`${API_BASE}/api/voice/stt`, { method: 'POST', body: fd });
    const j = await r.json();
    const text = (j.transcript || '').trim();
    if (text) {
      log('success', 'STT', { transcript: text });
      $('#input-text').value = text;
      await sendMessage(text);
    } else {
      log('system', 'STT', j.reason ? `no speech (${j.reason})` : 'no speech detected');
      // In conversation mode, just listen again instead of stopping.
      if (state.conversationMode) {
        state.emptyTurns = (state.emptyTurns || 0) + 1;
        if (state.emptyTurns >= 3) {
          pushBubble('system', 'No speech heard 3 times — conversation paused. Tap the orb to resume.');
          endConversation();
        } else {
          setOrb('listening', 'still listening…');
          await wait(400);
          startListening();
        }
      } else {
        pushBubble('system', 'No speech detected — try again or type.');
        afterTurn();
      }
    }
  } catch (e) {
    log('error', 'STT request failed', String(e).slice(0, 120));
    afterTurn();
  }
}

// ---- browser path (webkitSpeechRecognition) ----
function startBrowserListening() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    pushBubble('system', '⚠ Voice input unavailable. Install server STT (vosk) or use Chrome. Type instead.');
    return;
  }
  const r = new SR();
  r.lang = 'te-IN';
  r.continuous = false;
  r.interimResults = true;
  r.maxAlternatives = 1;
  r.onstart = () => {
    state.listening = true;
    setOrb('listening', 'వింటున్నాను…');
    $('#mic-btn').classList.add('danger');
    $('#mic-btn').textContent = '⏹';
    updateVoicePill();
  };
  r.onresult = (ev) => {
    let interim = '', final = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const alt = ev.results[i][0];
      if (ev.results[i].isFinal) final += alt.transcript;
      else interim += alt.transcript;
    }
    if (interim) {
      $('#input-text').value = interim;
      setOrb('listening', `వింటున్నాను: "${interim.slice(0, 30)}…"`);
    }
    if (final) {
      $('#input-text').value = final.trim();
      stopListening();
      sendMessage(final.trim());
    }
  };
  r.onerror = (ev) => {
    log('error', 'browser STT error', ev.error || 'unknown');
    if (ev.error === 'network' || ev.error === 'service-not-allowed') {
      pushBubble('system', 'Browser speech recognition unavailable — using server STT is recommended.');
    }
    stopListening();
  };
  r.onend = () => { if (state.listening) stopListening(); };
  try { r.start(); state.recognition = r; }
  catch (e) { log('error', 'STT start failed', e.message); stopListening(); }
}

/** Stop capturing (to transcribe). Does NOT end the conversation. */
function stopListening() {
  if (state.silenceTimer) { clearInterval(state.silenceTimer); state.silenceTimer = null; }
  if (state.recorder && state.recorder.state !== 'inactive') {
    try { state.recorder.stop(); } catch {}
    state.listening = false;
    updateMicUi();
    return;
  }
  if (state.recognition) {
    try { state.recognition.stop(); } catch {}
    state.recognition = null;
  }
  state.listening = false;
  updateMicUi();
}

/** Called after a turn finishes. Restarts listening if in conversation mode. */
async function afterTurn() {
  // In realtime streaming mode the socket stays open and keeps listening;
  // server-side VAD detects the next turn, so nothing to restart.
  if (state.streaming) return;
  if (state.conversationMode && !state.paused) {
    await wait(120);
    if (state.conversationMode && !state.paused) {
      startListening();
      return;
    }
  }
  resetMicUi();
}

function resetMicUi() {
  if (state.silenceTimer) { clearInterval(state.silenceTimer); state.silenceTimer = null; }
  state.listening = false;
  setOrb('idle', state.conversationMode ? 'conversation ended' : 'tap to talk');
  updateMicUi();
  updateVoicePill();
}

function updateMicUi() {
  const btn = $('#mic-btn');
  const orb = $('#orb');
  if (state.conversationMode) {
    btn.classList.add('danger');
    btn.textContent = '⏹';
    btn.title = 'Stop conversation';
    orb.classList.add('convo');
  } else {
    btn.classList.remove('danger');
    btn.textContent = '🎤';
    btn.title = 'Start conversation';
    orb.classList.remove('convo');
  }
}

// ---------- Conversation mode (hands-free loop) ------------------------
async function startConversation() {
  state.conversationMode = true;
  state.paused = false;
  state.emptyTurns = 0;
  _unlockAudio();   // must run inside the user gesture
  updateMicUi();
  pushBubble('system', '🎙 Conversation mode ON — speak freely. Ruchi will keep listening after each reply. Tap ⏹ to stop.');
  // Ruchi stays silent until the user starts the conversation. On the
  // first tap, speak the mandatory greeting, then listen.
  if (!state.greeted) {
    state.greeted = true;
    const greeting = "హాయ్, నేను రుచి, మీ వంటింటి సఖిని. ఈ వేళ ఏం వండుదాం, చెప్పు";
    pushBubble('ai', greeting);
    await speak(greeting);
  }
  await startListening();
}

function endConversation() {
  state.conversationMode = false;
  state.paused = false;
  state.pendingBargeText = null;
  state.bargeIn = false;
  if (state.turnAbort) { try { state.turnAbort.abort(); } catch {} state.turnAbort = null; }
  _stopTts();
  stopStreaming();
  updateMicUi();
  resetMicUi();
}

function toggleConversation() {
  if (state.conversationMode) {
    pushBubble('system', 'Conversation ended.');
    endConversation();
  } else {
    startConversation();
  }
}

// ---------- Transcript viewer ------------------------------------------
async function loadTranscript() {
  const pre = $('#transcript-text');
  const meta = $('#transcript-meta');
  if (!pre) return;
  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/transcript`);
    const text = await r.text();
    pre.textContent = text || '(empty)';
    pre.scrollTop = pre.scrollHeight;
    try {
      const fr = await fetch(`${API_BASE}/api/voice-agent/transcript/files`);
      const fj = await fr.json();
      const files = (fj.files || []).map((f) => `${f.name} (${f.bytes}B)`).join(', ');
      meta.textContent = files ? `saved: ${files}` : 'no files yet';
    } catch { meta.textContent = ''; }
  } catch (e) {
    pre.textContent = 'Could not load transcript: ' + e.message;
  }
}

function updateVoicePill() {
  const p = $('#voice-pill');
  const serverStt = state.caps?.sarvam || (state.caps?.vosk && state.caps?.vosk_model && state.caps?.ffmpeg);
  const serverTts = state.caps?.elevenlabs || state.caps?.edge_tts;
  const browserStt = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  const browserTts = !!window.speechSynthesis;
  const inOk = serverStt || browserStt;
  const outOk = serverTts || browserTts;
  if (state.conversationMode) {
    p.textContent = '● conversation';
    p.classList.remove('dead'); p.classList.add('live');
  } else if (inOk && outOk) {
    p.textContent = serverStt && serverTts ? 'voice ✓ (server)' : 'voice ✓';
    p.classList.remove('dead'); p.classList.add('live');
  } else {
    p.textContent = `voice ${inOk ? 'in✓' : 'in✗'} ${outOk ? 'out✓' : 'out✗'}`;
    p.classList.remove('live'); p.classList.add('dead');
  }
}

// ---------- Session lifecycle -------------------------------------------
/** Stable per-browser identity so Ruchi remembers this person. */
function getUserId() {
  try {
    let id = localStorage.getItem('ruchi_user_id');
    if (!id) {
      id = 'u_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem('ruchi_user_id', id);
    }
    return id;
  } catch {
    return 'default';
  }
}

async function createSession(mode) {
  const body = { mode, user_id: getUserId() };
  const r = await fetch(`${API_BASE}/api/voice-agent/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`createSession ${r.status}`);
  const j = await r.json();
  state.sessionId = j.session_id;
  state.eventCount = 0;
  state.lastState = j;
  $('#session-id').textContent = j.session_id.slice(0, 8) + '…';
  const rec = $('#recording-btn');
  if (rec) rec.href = `${API_BASE}/api/voice-agent/sessions/${j.session_id}/recording`;
  $('#recipe-name').textContent = j.recipe?.recipeName || 'No recipe yet';
  renderRecipeSteps(j.recipe);
  renderIntakeFields(j);
  refreshState(j);
  pushBubble('system', `Session ready — ${mode} mode. ${j.recipe ? j.recipe.recipeName + ' loaded.' : ''}`);
  startPolling();
}

/** Swap the recipe mid-session (user named a new dish). */
async function setRecipe(recipe) {
  RECIPE = recipe;
  $('#recipe-name').textContent = recipe.recipeName;
  renderRecipeSteps({ recipeName: recipe.recipeName, steps: recipe.steps, stepIndex: 0 });
  $('#state-step').textContent = `1 / ${recipe.steps.length}`;
  flashTile('#state-step');
  pushBubble('system', `📖 Switched to ${recipe.recipeName} (${recipe.steps.length} steps).`);
  if (!state.sessionId) return;
  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/recipe`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipe: { ...recipe, stepIndex: 0 } }),
    });
    if (r.ok) {
      const j = await r.json();
      state.lastState = j;
    }
  } catch (e) {
    log('error', 'set recipe failed', String(e).slice(0, 100));
  }
}

async function resetSession() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  stopStreaming();
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`, { method: 'DELETE' }); } catch {}
  }
  state.sessionId = null;
  state.greeted = false;
  $('#transcript').innerHTML = '';
  $('#log').innerHTML = '';
  pushBubble('system', 'New session starting…');
  await createSession(state.mode);
  if (state.conversationMode) startListening();
}

// ---------- State (event-driven, instant) ------------------------------
let _timerHandle = null;

function refreshState(j) {
  state.lastState = j;
  $('#state-field').textContent = j.current_field || '—';
  $('#state-step').textContent =
    j.recipe ? `${j.recipe.stepIndex + 1} / ${j.recipe.steps.length}` : '—';
  $('#state-paused').textContent = j.paused ? 'yes' : 'no';
  renderRecipeSteps(j.recipe);
  renderIntakeFields(j);
  renderPausedVisual(j.paused);
  renderTimerTile(j.active_timer);
}

/** Apply only the *new* events since the last turn, instantly. */
function applyEvents(events) {
  if (!Array.isArray(events)) return;
  const fresh = events.slice(state.eventCount || 0);
  state.eventCount = events.length;
  fresh.forEach((ev) => applyEvent(ev));
  if (fresh.length) {
    // keep the session state in sync after applying
    refreshStateFromLocal();
  }
}

function applyEvent(ev) {
  const t = ev.type;
  switch (t) {
    case 'step_changed':
      if (state.lastState?.recipe) {
        state.lastState.recipe.stepIndex = ev.stepIndex;
        renderRecipeSteps(state.lastState.recipe);
      }
      $('#state-step').textContent = state.lastState?.recipe
        ? `${ev.stepIndex + 1} / ${state.lastState.recipe.steps.length}` : '—';
      flashTile('#state-step');
      break;
    case 'recipe_changed':
      state.lastState = state.lastState || {};
      state.lastState.recipe = {
        recipeName: ev.recipeName, steps: ev.steps, stepIndex: ev.stepIndex || 0,
      };
      RECIPE = { recipeName: ev.recipeName, steps: ev.steps };
      $('#recipe-name').textContent = ev.recipeName;
      renderRecipeSteps(state.lastState.recipe);
      flashTile('#state-step');
      break;
    case 'recipe_loaded':
      state.lastState = state.lastState || {};
      state.lastState.recipe = {
        recipeName: ev.recipeName, steps: ev.steps, stepIndex: ev.stepIndex || 0,
      };
      RECIPE = { recipeName: ev.recipeName, steps: ev.steps };
      $('#recipe-name').textContent = ev.recipeName;
      renderRecipeSteps(state.lastState.recipe);
      pushBubble('system', `📖 Loaded ${ev.recipeName} (${(ev.steps || []).length} steps).`);
      flashTile('#state-step');
      break;
    case 'servings_set':
      pushBubble('system', `👥 Scaled ingredients for ${ev.servings} ${ev.servings === 1 ? 'person' : 'people'}.`);
      break;
    case 'step_completed':
      pushBubble('system', `✅ Step ${(ev.stepIndex ?? 0) + 1} already done (parallel).`);
      break;
    case 'recipe_suggestions': {
      const names = (ev.candidates || []).map((c) => c.name).join(', ');
      if (names) pushBubble('system', `💡 Suggestions: ${names}`);
      break;
    }
    case 'timer_started':
      state.timerFired = false;
      startTimerCountdown(ev.seconds, ev.label);
      flashTile('#state-timer');
      break;
    case 'timer_cancelled':
      state.timerFired = false;
      stopTimerCountdown();
      break;
    case 'timer_expired':
      onTimerExpired(ev.label);
      break;
    case 'paused':
      state.paused = true;
      renderPausedVisual(true);
      break;
    case 'resumed':
      state.paused = false;
      renderPausedVisual(false);
      break;
    case 'issue_reported':
      state.lastState = state.lastState || {};
      state.lastState.active_issue = { issue: ev.issue, details: ev.details };
      flashTile('#state-field');
      break;
    case 'issue_resolved':
      if (state.lastState) state.lastState.active_issue = null;
      break;
    case 'field_saved':
      if (state.lastState) {
        state.lastState.answers = state.lastState.answers || {};
        state.lastState.answers[ev.field] = ev.value;
        state.lastState.current_field = ev.next_field || '';
        renderIntakeFields(state.lastState);
        $('#state-field').textContent = ev.next_field || '—';
        flashTile('#state-field');
      }
      break;
    case 'interview_complete':
      pushBubble('system', '✅ Recipe intake complete — all fields saved.');
      break;
    case 'cooking_finished':
      pushBubble('system', '🍽 Cooking finished.');
      stopTimerCountdown();
      break;
    default:
      break;
  }
}

function refreshStateFromLocal() {
  const j = state.lastState;
  if (!j) return;
  $('#state-field').textContent = j.current_field || '—';
  if (j.recipe) {
    $('#state-step').textContent = `${j.recipe.stepIndex + 1} / ${j.recipe.steps.length}`;
    renderRecipeSteps(j.recipe);
  }
  renderIntakeFields(j);
}

function flashTile(sel) {
  const el = $(sel)?.closest('.state-tile');
  if (!el) return;
  el.classList.remove('flash');
  void el.offsetWidth;   // restart animation
  el.classList.add('flash');
  setTimeout(() => el.classList.remove('flash'), 900);
}

function renderPausedVisual(paused) {
  $('#state-paused').textContent = paused ? 'yes' : 'no';
  const tile = $('#state-paused')?.closest('.state-tile');
  tile?.classList.toggle('alert', !!paused);
  $('#orb').classList.toggle('paused', !!paused);
}

// ---------- Live timer countdown ---------------------------------------
function renderTimerTile(t) {
  if (!t) { $('#state-timer').textContent = '—'; return; }
  // if a countdown is already running, don't clobber it
  if (state.timerEndsAt) return;
  $('#state-timer').textContent = `${t.seconds}s · ${t.label}`;
}

function startTimerCountdown(seconds, label) {
  stopTimerCountdown();
  state.timerEndsAt = Date.now() + seconds * 1000;
  state.timerLabel = label;
  const tick = () => {
    const remain = Math.max(0, Math.round((state.timerEndsAt - Date.now()) / 1000));
    const mm = String(Math.floor(remain / 60)).padStart(2, '0');
    const ss = String(remain % 60).padStart(2, '0');
    $('#state-timer').textContent = `${mm}:${ss} · ${label}`;
    if (remain <= 0) { stopTimerCountdown(); onTimerExpired(label); return; }
    state.timerHandle = setTimeout(tick, 250);
  };
  tick();
}

function stopTimerCountdown() {
  if (state.timerHandle) { clearTimeout(state.timerHandle); state.timerHandle = null; }
  state.timerEndsAt = null;
  state.timerLabel = null;
  $('#state-timer').textContent = '—';
}

function onTimerExpired(label) {
  stopTimerCountdown();
  // The client countdown and the server `timer_expired` event both land
  // here — only act once.
  if (state.timerFired) return;
  state.timerFired = true;
  pushBubble('system', `⏰ Time's up — check the food.`);
  const tile = $('#state-timer')?.closest('.state-tile');
  if (tile) {
    tile.classList.add('alert');
    setTimeout(() => tile.classList.remove('alert'), 3000);
  }
  // chime if possible
  try {
    const ac = new (window.AudioContext || window.webkitAudioContext)();
    const o = ac.createOscillator(); const g = ac.createGain();
    o.connect(g); g.connect(ac.destination);
    o.frequency.value = 880; g.gain.value = 0.15;
    o.start(); o.stop(ac.currentTime + 0.5);
  } catch {}
  // Automatically advance to the next step (Ruchi will speak it).
  _advanceAfterTimer();
}

/** Ask Ruchi for the next step once the timer is done. Retries if a turn
 *  is already in flight (e.g. she's still speaking). */
async function _advanceAfterTimer(attempt = 0) {
  if (state.paused || state.mode !== 'cook') return;
  if (state.busy) {
    if (attempt < 12) setTimeout(() => _advanceAfterTimer(attempt + 1), 700);
    return;
  }
  sendMessage('టైమర్ అయిపోయింది. వంట సిద్ధమైందో చెక్ చేయమని చెప్పి, తర్వాత స్టెప్ చెప్పు.', { silent: true });
}

// Slow background poll as a safety net only (events give instant updates).
function startPolling() {
  if (state.pollHandle) clearInterval(state.pollHandle);
  state.pollHandle = setInterval(async () => {
    if (!state.sessionId) return;
    try {
      const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`);
      if (!r.ok) return;
      const j = await r.json();
      refreshState(j);
      state.eventCount = (j.events || []).length;
    } catch {}
  }, 5000);
}

// ---------- Recipe steps render -----------------------------------------
function renderRecipeSteps(recipe) {
  const ol = $('#recipe-steps');
  ol.innerHTML = '';
  if (!recipe) return;
  recipe.steps.forEach((s, i) => {
    const li = document.createElement('li');
    if (i < recipe.stepIndex) li.className = 'done';
    else if (i === recipe.stepIndex) li.className = 'active';
    const num = document.createElement('span');
    num.className = 'num';
    num.textContent = i + 1;
    li.appendChild(num);
    li.appendChild(document.createTextNode(s));
    ol.appendChild(li);
  });
}

// ---------- Intake fields render ----------------------------------------
function renderIntakeFields(j) {
  const card = $('#intake-card');
  const list = $('#intake-fields');
  if (state.mode !== 'intake') { card.style.display = 'none'; return; }
  card.style.display = '';
  list.innerHTML = '';
  INTAKE_FIELDS.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'intake-field';
    const val = j.answers?.[f.name];
    const isCurrent = !val && j.current_field === f.name;
    const isDone = !!val;
    if (isDone) row.classList.add('done');
    else if (isCurrent) row.classList.add('current');
    row.innerHTML = `
      <span class="check">${isDone ? '✓' : (isCurrent ? '…' : '')}</span>
      <span class="name">${f.name}</span>
      <span class="val">${val || ''}</span>
    `;
    list.appendChild(row);
  });
}

// ---------- Conversation / log ------------------------------------------
/** Remove ElevenLabs emotion tags (e.g. [warm]) for on-screen display.
 *  The tags are still sent to TTS, which uses them for delivery. */
function stripTags(text) {
  return (text || '').replace(/\[\s*[A-Za-z]+(?:\s*,\s*[A-Za-z]+)*\s*\]/g, ' ')
    .replace(/\s{2,}/g, ' ').trim();
}

function pushBubble(kind, text) {
  const t = $('#transcript');
  // Remove empty state if present
  const empty = t.querySelector('.transcript-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = `bubble ${kind}`;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = kind === 'user' ? 'You' : kind === 'ai' ? 'Ruchi' : '·';
  const body = document.createElement('div');
  body.className = 'text';
  body.textContent = text;
  div.append(who, body);
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
  return div;
}

function pushToolAnnotation(toolName, ok = true) {
  const t = $('#transcript');
  const empty = t.querySelector('.transcript-empty');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = `tool-annotation ${ok ? '' : 'error'}`;
  div.innerHTML = `<span class="dot"></span> tool fired: <b>${toolName}</b>`;
  t.appendChild(div);
  t.scrollTop = t.scrollHeight;
}

function log(kind, name, payload) {
  const entry = document.createElement('div');
  entry.className = `log-entry ${kind}`;
  const head = document.createElement('div');
  head.className = 'head';
  const nameEl = document.createElement('span');
  nameEl.className = 'name';
  nameEl.textContent = name;
  const timeEl = document.createElement('span');
  timeEl.className = 'time';
  timeEl.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  head.append(nameEl, timeEl);
  entry.append(head);
  if (payload !== undefined) {
    const pre = document.createElement('pre');
    pre.textContent = typeof payload === 'string' ? payload : JSON.stringify(payload, null, 2);
    entry.append(pre);
  }
  const logEl = $('#log');
  logEl.appendChild(entry);
  logEl.scrollTop = logEl.scrollHeight;
}

// ---------- Orb state ----------------------------------------------------
function setOrb(stateName, label) {
  state.orbState = stateName;
  const orb = $('#orb');
  const status = $('#orb-status');
  orb.classList.remove('idle', 'listening', 'thinking', 'speaking', 'tool');
  if (stateName !== 'idle') orb.classList.add(stateName);
  else orb.classList.add('idle');
  status.classList.remove('listening', 'thinking', 'speaking', 'tool');
  if (stateName !== 'idle') status.classList.add(stateName);
  status.textContent = label || stateName;
}

// ---------- Message send ------------------------------------------------
async function sendMessage(text, { silent = false } = {}) {
  if (!text.trim()) return;
  if (state.busy) {
    // Barge-in while a turn is in flight — queue and cancel the old turn.
    state.pendingBargeText = text;
    _stopTts();
    if (state.turnAbort) { try { state.turnAbort.abort(); } catch {} }
    return;
  }
  state.busy = true;
  state.bargeIn = false;
  state.turnAbort = new AbortController();
  const signal = state.turnAbort.signal;
  $('#input-text').value = '';
  if (!silent) pushBubble('user', text);
  setOrb('thinking', 'thinking…');

  try {
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/turn`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal,
    });
    const j = await r.json();

    // Apply state changes from events immediately (no polling lag).
    applyEvents(j.events);

    // Annotate every tool that fired
    (j.tool_calls || []).forEach((tc, i) => {
      const tr = (j.tool_results || [])[i] || {};
      const ok = !tr.error;
      pushToolAnnotation(tc.name, ok);
      log(ok ? 'success' : 'error', `${ok ? '←' : '✗'} ${tc.name}`, {
        args: tc.args,
        result: tr.result,
        error: tr.error,
      });
      // The `pause` tool means "stop talking to me" — stop the loop.
      if (tc.name === 'pause') state.paused = true;
      // `resume` clears the pause and lets the loop continue.
      if (tc.name === 'resume') state.paused = false;
      // `finish_cooking` ends the session once the dish is done.
      if (tc.name === 'finish_cooking') state.finishRequested = true;
    });

    if (j.error) {
      pushBubble('system', `⚠ ${j.error}`);
      log('error', 'turn failed', j.error);
    }
    if (j.assistant_text && !state.pendingBargeText) {
      pushBubble('ai', j.assistant_text);
      await speak(j.assistant_text);   // wait until Ruchi finishes speaking
    }
    refreshStateFromLocal();
  } catch (err) {
    if (err.name !== 'AbortError') {
      pushBubble('system', `⚠ ${err.message}`);
      log('error', 'request failed', err.message);
    }
  } finally {
    state.busy = false;
    state.turnAbort = null;
    state.speaking = false;
    state.aiText = '';
    clearTimeout(state.bargeWatchdog);
    state.bargeWatchdog = null;

    // Barge-in: the user interrupted — answer their new query instead.
    const pending = state.pendingBargeText;
    state.pendingBargeText = null;
    if (pending) {
      state.bargeIn = false;
      sendMessage(pending);
      return;
    }
    // The model signalled the cooking is finished. Ruchi NEVER ends the
    // conversation on her own — she asks for feedback and keeps talking.
    if (state.finishRequested) {
      state.finishRequested = false;
      pushBubble('system', '🍽 Cooking finished — Ruchi will keep chatting.');
    }
    if (state.paused) {
      pushBubble('system', 'Paused — tap the orb to resume the conversation.');
      endConversation();
      return;
    }
    // Otherwise keep the hands-free conversation going until cooking is done.
    await afterTurn();
  }
}

async function getState() {
  const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`);
  return r.json();
}

// ---------- Direct tool invocation (chips) -----------------------------
async function callToolDirect(toolName, args, btn) {
  if (state.busy) return;
  state.busy = true;

  // Instant visual feedback on the chip itself.
  if (btn) {
    btn.classList.add('firing');
    setTimeout(() => btn.classList.remove('firing'), 700);
  }
  setOrb('tool', `→ ${toolName}`);
  log('system', `→ ${toolName}`, args);

  try {
    // Every tool goes through the direct-call endpoint so pressing
    // "start_timer" fires exactly start_timer — no LLM in the loop.
    const r = await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}/call`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: toolName, args }),
    });
    const j = await r.json();

    if (!r.ok || j.detail) {
      pushToolAnnotation(toolName, false);
      log('error', `✗ ${toolName}`, j.detail || j);
      return;
    }

    pushToolAnnotation(toolName, true);
    log('success', `← ${toolName}`, j.result);

    // Apply the resulting events instantly (step change, timer, pause…).
    applyEvents(j.events);

    // Speak the canned phrase if the handler produced one.
    const phrase = j.result?.phrase || j.result?.lead_in;
    if (phrase) {
      pushBubble('ai', phrase);
      await speak(phrase);
    }

    // Keep the full session snapshot in sync.
    if (j.session) refreshState(j.session);
  } catch (err) {
    pushToolAnnotation(toolName, false);
    log('error', `← ${toolName}`, err.message);
  } finally {
    state.busy = false;
    setTimeout(() => {
      if (state.orbState === 'tool') setOrb('idle', 'tap to talk');
    }, 600);
  }
}

function synthesize(toolName, args) {
  switch (toolName) {
    case 'finish_interview': return 'finish';
    case 'next_step': return 'okay next';
    case 'previous_step': return 'go back';
    case 'goto_step': return `go to step ${args.step}`;
    case 'start_timer': return `set a timer for ${args.seconds} seconds`;
    case 'cancel_timer': return 'cancel the timer';
    case 'pause': return 'pause.';
    case 'resume': return 'continue';
    case 'report_issue': return `this is too ${args.issue}`;
    case 'resolve_issue': return 'fixed, thanks';
    case 'finish_cooking': return 'finish cooking';
    default: return toolName;
  }
}

function gatherArgs(tool) {
  switch (tool) {
    case 'save_answer':
      return {
        field: $('#intake-field').value.trim(),
        value: $('#intake-value').value.trim(),
      };
    case 'goto_step': {
      const n = parseInt($('#goto-step').value, 10);
      if (!Number.isFinite(n) || n < 1) {
        log('error', 'bad arg', { tool, reason: 'step must be ≥ 1' });
        return null;
      }
      return { step: n };
    }
    case 'start_timer': {
      const s = parseInt($('#timer-seconds').value, 10);
      if (!Number.isFinite(s) || s < 1) {
        log('error', 'bad arg', { tool, reason: 'seconds must be ≥ 1' });
        return null;
      }
      return { seconds: s, label: $('#timer-label').value.trim() || 'your timer' };
    }
    case 'report_issue':
      return { issue: $('#issue-cat').value, details: $('#issue-details').value.trim() };
    default:
      return {};
  }
}

// ---------- Mode switching ----------------------------------------------
async function switchMode(mode) {
  if (mode === state.mode) return;
  state.mode = mode;
  $('#mode-intake').classList.toggle('active', mode === 'intake');
  $('#mode-cook').classList.toggle('active', mode === 'cook');
  $('#group-intake').style.display = mode === 'intake' ? '' : 'none';
  stopStreaming();
  if (state.sessionId) {
    try { await fetch(`${API_BASE}/api/voice-agent/sessions/${state.sessionId}`, { method: 'DELETE' }); } catch {}
  }
  state.sessionId = null;
  $('#transcript').innerHTML = '';
  await createSession(mode);
  if (state.conversationMode) startListening();
  // Speak the first intake question
  if (mode === 'intake') {
    const q = INTAKE_FIELDS[0].question;
    pushBubble('ai', q);
    speak(q);
  }
  // Cook mode stays silent until the user taps the orb / mic button.
}

// ---------- UI wiring ----------------------------------------------------
function wireUi() {
  $('#mode-intake').addEventListener('click', () => switchMode('intake'));
  $('#mode-cook').addEventListener('click', () => switchMode('cook'));

  $('#input-form').addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage($('#input-text').value);
  });

  // Mic button toggles hands-free conversation mode on/off.
  $('#mic-btn').addEventListener('click', () => {
    toggleConversation();
  });

  // Orb does the same.
  $('#orb').addEventListener('click', () => {
    toggleConversation();
  });

  $('#reset-btn').addEventListener('click', resetSession);

  // Transcript panel
  $('#transcript-refresh')?.addEventListener('click', loadTranscript);
  $('#transcript-clear')?.addEventListener('click', async () => {
    if (!confirm('Clear all transcripts on the server?')) return;
    try {
      await fetch(`${API_BASE}/api/voice-agent/transcript`, { method: 'DELETE' });
      await loadTranscript();
    } catch (e) { log('error', 'clear transcript failed', String(e)); }
  });

  $$('.side-tabs button').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.side-tabs button').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.tab;
      $$('.tab-panel').forEach((p) => p.classList.toggle('active', p.dataset.panel === target));
      if (target === 'transcript') loadTranscript();
    });
  });

  $$('.chip[data-tool]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const tool = btn.dataset.tool;
      const args = gatherArgs(tool);
      if (args === null) return;
      await callToolDirect(tool, args, btn);
    });
  });

  // Keyboard: enter to send
  $('#input-text').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage($('#input-text').value);
    }
  });
}

// ---------- Boot --------------------------------------------------------
(async function boot() {
  await loadCaps();
  // Only touch the browser SpeechSynthesis API when we actually need it.
  // On Linux Firefox, calling speechSynthesis.getVoices() logs
  // "Speech Dispatcher required for SpeechSynthesis API" when
  // speech-dispatcher isn't installed. Skip it entirely when server-side
  // TTS (ElevenLabs / edge-tts) is available.
  if (!state.caps?.elevenlabs && !state.caps?.edge_tts) {
    await loadVoices();
  }
  wireUi();
  await ping();
  setInterval(ping, 5000);
  updateVoicePill();
  await createSession('cook');

  // Diagnostics: tell the user exactly which providers are active.
  const c = state.caps || {};
  const inMode = c.sarvam ? `Sarvam ${c.sarvam_model || 'saaras:v3'} (cloud)` :
                 ((c.vosk && c.vosk_model && c.ffmpeg) ? 'Vosk (offline Telugu)' :
                  ((window.SpeechRecognition || window.webkitSpeechRecognition) ? 'browser Web Speech' : 'none'));
  const outMode = c.elevenlabs ? `ElevenLabs ${c.elevenlabs_model || 'eleven_v3_conversational'}` :
                  (c.edge_tts ? 'edge-tts (Telugu neural)' :
                   (window.speechSynthesis ? 'browser speechSynthesis' : 'none'));
  const llmMode = c.gemini ? `Gemini ${c.gemini_model || ''}` : 'mock runner (no GOOGLE_API_KEY)';
  pushBubble('system', `Voice in: ${inMode} · Voice out: ${outMode} · LLM: ${llmMode}`);
  if (inMode === 'none') {
    pushBubble('system', '⚠ No voice input available — type your messages.');
  }
  if (outMode === 'none') {
    pushBubble('system', '⚠ No voice output available.');
  }

  // Ruchi stays silent until the user taps the orb / mic button.
  pushBubble('system', 'Tap the 🎤 button or the orb to start talking with Ruchi.');
})();
