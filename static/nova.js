// Nova web client (served by app.py on this computer).
// Mic → 16 kHz PCM16 frames → /ws/voice; Nova's sentences come back as WAV + a "sentence" caption event
// and play gaplessly. Every number shown is measured live. No network resources are loaded.
(() => {
  const $ = (id) => document.getElementById(id);
  const pageLabel = (pg) => (pg == null || pg === '' ? '' : (/^\d+$/.test(String(pg)) ? 'p. ' + pg : String(pg)));
  let templates = [], tmpl = null;
  const qs = new URLSearchParams(location.search);
  let profile = qs.get('profile') || 'nova', voice = qs.get('voice') || '';
  let profiles = [], voices = [], pipeInfo = null;
  let ws = null, capCtx = null, playCtx = null, analyser = null, micStream = null, worklet = null;
  let nextStart = 0, sources = [], micLevel = 0, muted = false, uiState = 'idle', playGen = 0, playQ = Promise.resolve();
  let outGain = null, timerEnd = 0, timerLabel = '';
  let lastSpeechT = null, speechEnd = null, firstAudioT = null, turnNo = 0;
  const pendingCaptions = [];
  let replySentences = [], recalled = new Set(), learned = new Set(), connId = 0;

  // ---------------- helpers ----------------
  const fmtS = (ms) => (ms / 1000).toFixed(2) + ' s';
  const r0 = (v) => (v == null ? '–' : Math.round(v));
  function toast(msg) { const t = $('toast'); t.textContent = msg; t.classList.remove('hidden'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add('hidden'), 2200); }
  function nice(fact) { // third-person store text → second person for the audience
    return fact.replace(/\.$/, '')
      .replace(/^User's name is /, 'Your name is ')
      .replace(/^User asked to be reminded to /, 'Remind you to ')
      .replace(/^User is /, "You're ").replace(/^User has /, 'You have ')
      .replace(/^User's /, 'Your ').replace(/^User (\w+?)s\b/, 'You $1').replace(/^User /, 'You ');
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  // ---------------- network badge (real interface state from the server) ----------------
  async function pollNet() {
    try {
      const { data } = await (await fetch('/api/offline')).json();
      const off = data.offline;
      for (const [dot, txt] of [['start-dot', 'start-net'], ['net-dot', 'net-text']]) {
        $(dot).className = 'dot ' + (off ? 'on' : 'off');
        $(txt).textContent = off ? 'Offline · everything on this computer' : 'Online · only web tools use the internet';
      }
    } catch (_) { /* server restarting */ }
  }
  pollNet(); setInterval(pollNet, 5000);

  // ---------------- start screen ----------------
  async function loadStart() {
    try { profiles = (await (await fetch('/api/memory/profiles')).json()).data.profiles; } catch (_) { profiles = [{ id: profile, title: 'Nova', facts: 0 }]; }
    if (!profiles.some((p) => p.id === profile) && profiles.length) profile = profiles[0].id;
    let def = 'ljspeech';
    try { const d = (await (await fetch('/api/voices/open')).json()).data; voices = d.voices || []; def = d.default || def; } catch (_) { voices = [{ id: 'ljspeech', display: 'LJSpeech' }]; }
    if (!voices.some((v) => v.id === voice)) voice = def;
    try { pipeInfo = (await (await fetch('/api/pipeline')).json()).data; } catch (_) { pipeInfo = null; }
    if (pipeInfo && pipeInfo.llm) $('hw').textContent = `Language model: ${pipeInfo.llm.model} · ${pipeInfo.llm.runtime}`;
    const vc = $('voices'); vc.innerHTML = '';
    for (const v of voices) {
      const b = document.createElement('button');
      b.textContent = (v.display || v.id).split(' (')[0];
      b.title = v.display || v.id;
      b.setAttribute('aria-pressed', String(v.id === voice));
      b.onclick = () => { voice = v.id; [...vc.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b))); };
      vc.appendChild(b);
    }
    loadTemplates();
  }
  // ---------------- use-case templates (templates.py): saved per person ----------------
  async function loadTemplates() {
    try {
      const { data } = await (await fetch(`/api/templates?profile=${profile}`)).json();
      templates = data.templates; tmpl = data.current;
    } catch (_) { templates = []; tmpl = null; }
    const tc = $('templates'); if (!tc) return; tc.innerHTML = '';
    for (const t of templates) {
      const b = document.createElement('button');
      b.textContent = t.name; b.title = t.blurb; b.setAttribute('aria-pressed', String(t.id === tmpl));
      b.onclick = async () => {
        const r = await (await fetch('/api/templates', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ profile, template: t.id }) })).json();
        if (!r.ok) { toast(r.error ? r.error.message : 'Could not change the template'); return; }
        tmpl = t.id; [...tc.children].forEach((c) => c.setAttribute('aria-pressed', String(c === b)));
        $('tmpl-blurb').textContent = t.blurb;
      };
      tc.appendChild(b);
    }
    const cur = templates.find((t) => t.id === tmpl); $('tmpl-blurb').textContent = cur ? cur.blurb : '';
  }
  loadStart();
  $('begin').onclick = begin;

  async function begin() {
    $('begin').disabled = true;
    try {
      playCtx = playCtx || new AudioContext();
      if (playCtx.state === 'suspended') await playCtx.resume();
      if (!analyser) { analyser = playCtx.createAnalyser(); analyser.fftSize = 512; outGain = playCtx.createGain(); analyser.connect(outGain); outGain.connect(playCtx.destination); }
      if (!micStream) await startMic();
    } catch (e) {
      toast('Microphone unavailable — allow microphone access and try again.');
    }
    $('start').classList.add('hidden'); $('call').classList.remove('hidden');
    $('begin').disabled = false;
    setUrl(); connect(true); renderTop(); loadMemory(); renderPipe();
  }

  function setUrl() { const u = new URLSearchParams(location.search); u.set('profile', profile); u.set('voice', voice); history.replaceState(null, '', '?' + u); }

  async function startMic() {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: false, sampleRate: 16000 } });
    capCtx = new AudioContext({ sampleRate: 16000 });
    await capCtx.audioWorklet.addModule('pcm-worklet.js');
    const src = capCtx.createMediaStreamSource(micStream);
    worklet = new AudioWorkletNode(capCtx, 'pcm16-frames');
    const g = capCtx.createGain(); g.gain.value = 0;
    src.connect(worklet); worklet.connect(g); g.connect(capCtx.destination);
    worklet.port.onmessage = ({ data }) => onFrame(data.pcm, data.peak);
  }

  const SILENT = new Int16Array(512).buffer;
  function onFrame(pcm, peak) {
    if (muted) { pcm = SILENT.slice(0); peak = 0; }
    if (peak > 0.03) lastSpeechT = performance.now();
    micLevel = micLevel * 0.7 + peak * 0.3;
    if (ws && ws.readyState === 1) ws.send(pcm);
  }

  // ---------------- websocket ----------------
  function connect(greet) {
    const id = ++connId;
    if (ws) { try { ws.onclose = null; ws.close(); } catch (_) {} }
    stopPlayback(); clearCall();
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/voice`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'config', voice, language: 'en', profile }));
      if (greet) setTimeout(() => { if (id === connId && ws.readyState === 1) ws.send(JSON.stringify({ type: 'greet' })); }, 250);
      setState('listening');
    };
    ws.onclose = () => { if (id === connId) { setState('idle'); toast('Connection closed — press Esc and start again.'); } };
    ws.onmessage = async (ev) => {
      if (id !== connId) return;
      if (ev.data instanceof ArrayBuffer) { const g = playGen; playQ = playQ.then(() => playChunk(ev.data, g)).catch(() => {}); return; }
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      switch (m.type) {
        case 'state':
          if (m.value === 'processing') { $('toolcard').classList.add('hidden'); leadNext = false; speechEnd = lastSpeechT; firstAudioT = null; replySentences = []; $('chips').innerHTML = ''; $('source').classList.add('hidden'); recalled = new Set(); learned = new Set(); }
          setState(m.value); break;
        case 'transcript': $('you').classList.remove('partial'); $('you').innerHTML = `<b>You:</b> ${esc(m.text)}`; break;
        case 'partial': if (uiState === 'listening' || uiState === 'interrupted') { $('you').classList.add('partial'); $('you').innerHTML = `<b>You:</b> ${esc(m.text)}`; } break;
        case 'stage': onStage(m); break;
        case 'sources': showSources(m.items || []); leadNext = !!m.leadin; break;
        case 'safety': $('activity').textContent = 'Child-safety guard gave a safe answer · nothing saved'; break;
        case 'tool': showTool(m); break;
        case 'device': onDevice(m); break;
        case 'timer_done': timerEnd = 0; renderTimer(); toast(`Timer done${m.label ? ': ' + m.label : ''}`); break;
        case 'memory': showRecalled(m.facts || []); break;
        case 'memory_saved': showLearned(m.facts || []); break;
        case 'sentence': pendingCaptions.push(m.text); break;
        case 'reply': break; // captions are driven by the per-sentence events, in time with the audio
        case 'barge_in': stopPlayback(); setState('interrupted'); $('said').innerHTML = `<span class="old">${replySentences.map(esc).join(' ')} —</span>`; break;
        case 'trace': onTrace(m); break;
        case 'audio_end': break;
        case 'error': toast('Something went wrong on the server: ' + m.message); break;
      }
    };
  }

  // ---------------- playback + timed captions ----------------
  async function playChunk(buf, gen) {
    const ctx = playCtx; if (!ctx || gen !== playGen) return;
    let audio;
    try { audio = await ctx.decodeAudioData(buf.slice(0)); } catch (_) { if (gen === playGen) pendingCaptions.shift(); return; } // keep captions aligned
    if (gen !== playGen) return; // a barge-in or reconnect happened while this chunk was decoding
    if (ctx.state === 'suspended') await ctx.resume();
    const t = Math.max(ctx.currentTime + 0.02, nextStart);
    const s = ctx.createBufferSource(); s.buffer = audio; s.connect(analyser); s.start(t);
    nextStart = t + audio.duration; sources.push(s);
    s.onended = () => { sources = sources.filter((x) => x !== s); if (!sources.length && uiState === 'speaking') setState('listening'); };
    const text = pendingCaptions.shift();
    const delay = Math.max(0, (t - ctx.currentTime) * 1000);
    if (firstAudioT === null) firstAudioT = performance.now() + delay;
    setTimeout(() => { if (sources.includes(s)) { if (text) caption(text); setState('speaking'); } }, delay);
  }
  let leadNext = false;
  function caption(text) {
    replySentences.push(text);
    const prev = replySentences.slice(0, -1).slice(-1);
    // the first sentence of a library answer is Nova's lead-in, spoken while the passage is read
    const tag = leadNext ? '<span class="tag">said while searching</span> ' : ''; leadNext = false;
    $('said').innerHTML = (prev.length ? `<span class="old">${esc(prev[0])}</span> ` : '') + tag + `<span class="now">${esc(text)}</span>`;
  }
  function stopPlayback() { playGen++; for (const s of sources) { try { s.stop(); } catch (_) {} } sources = []; nextStart = 0; pendingCaptions.length = 0; }
  // ---------------- live steps ----------------
  const STEP_TEXT = {
    hear: { start: 'Listening to you' },
    turn: { start: 'Checking if you have finished', wait: 'Waiting, you may not be done yet', done: 'Got it' },
    stt: { start: 'Transcribing what you said', done: 'Transcribed' },
    memory: { start: 'Searching memory', done: 'Memory checked' },
    rag: { start: () => `Looking through your ${libName}`, done: 'Library checked' },
    llm: { start: 'Nova is thinking', first: 'Nova started answering', done: 'First sentence ready' },
    tts: { start: 'Turning it into speech', done: 'Speaking' },
  };
  let stepsT0 = 0, libName = 'documents';
  function resetSteps() { document.querySelectorAll('#steps li').forEach((li) => { li.className = ''; li.querySelector('.sv').textContent = ''; }); $('activity').textContent = ''; }
  function onStage(m) {
    const li = document.querySelector(`#steps li[data-k="${m.stage}"]`); if (!li) return;
    const sv = li.querySelector('.sv');
    if (m.stage === 'hear' && m.status === 'start') { resetSteps(); stepsT0 = performance.now(); $('you').classList.add('partial'); $('you').textContent = ''; }
    if (m.status === 'start') { li.className = 'active'; sv.textContent = ''; }
    else if (m.status === 'wait') { li.className = 'wait'; sv.textContent = 'checking'; }
    else if (m.status === 'first') { li.className = 'active'; sv.textContent = `${m.ms} ms`; }
    else if (m.status === 'done') {
      li.className = 'done';
      sv.textContent = m.stage === 'memory' ? `${m.hits} found · ${m.ms} ms` : (m.ms != null ? `${m.ms} ms` : 'done');
      if (m.stage === 'turn') document.querySelector('#steps li[data-k="hear"]').className = 'done';
    }
    const t = STEP_TEXT[m.stage] && STEP_TEXT[m.stage][m.status];
    if (t) $('activity').textContent = (typeof t === 'function' ? t() : t) + (m.status === 'done' || m.status === 'first' ? (m.ms != null && m.stage !== 'memory' ? ` · ${m.ms} ms` : '') : '…');
    if (m.stage === 'rag' && m.status === 'start' && m.library) libName = m.library;
    if (m.stage === 'rag' && m.status === 'done') { sv.textContent = m.gate ? `${m.hits} found · ${m.ms} ms` : (m.ms ? `no match · ${m.ms} ms` : '–'); $('activity').textContent = m.gate ? `Found it in your ${libName} · ${m.ms} ms` : (m.reason ? 'Answering from memory' : `No match in your ${libName} · ${m.ms} ms`); }
    if (m.stage === 'tool' && m.status === 'start') { sv.textContent = m.title || ''; $('activity').textContent = `Using the ${m.title || 'tool'}${m.online ? ' (internet)' : ''}…`; }
    if (m.stage === 'tool' && m.status === 'done') { sv.textContent = `${m.ok ? '' : 'failed · '}${Math.round(m.ms || 0)} ms`; $('activity').textContent = `${m.ok ? 'Got the result' : 'The tool did not work'} · ${Math.round(m.ms || 0)} ms`; }
    if (m.stage === 'memory' && m.status === 'done') $('activity').textContent = m.hits ? `Found ${m.hits} ${m.hits === 1 ? 'memory' : 'memories'} in ${m.ms} ms` : `Nothing relevant in memory · ${m.ms} ms`;
    if (m.stage === 'stt' && m.status === 'done' && m.text) { $('you').classList.add('partial'); $('you').innerHTML = `<b>You:</b> ${esc(m.text)}`; }
    // mirror into the Details drawer cards
    const card = { hear: 'vad', turn: 'turn', stt: 'stt', memory: 'mem', rag: 'rag', tool: 'tool', llm: 'llm', tts: 'tts' }[m.stage];
    const el = document.querySelector(`#pipe .st[data-k="${card}"]`);
    if (el) { el.classList.toggle('active', m.status === 'start' || m.status === 'first'); el.classList.toggle('done', m.status === 'done'); }
  }

  // ---------------- tools ----------------
  const TOOL_ICON = { get_datetime: '🕒', timer: '⏱', reminders: '🔔', calendar: '📅', notes: '📝', calculate: '🧮', convert_units: '📏',
    forget_memory: '🧠', device_control: '🎚', search_library: '📚', get_weather: '🌤', web_search: '🔎', wikipedia: '📖', get_news: '📰', currency_convert: '💱' };
  function showTool(m) {
    const f = m.fields || {};
    const rows = Object.entries(f).filter(([k]) => k !== 'source' && k !== 'expression').slice(0, 5)
      .map(([k, v]) => `<span class="tk">${esc(k)}</span> ${esc(v)}`).join('<br>');
    const badge = m.online ? '<span class="tag net">Used the internet' + (m.source ? ' · ' + esc(m.source) : '') + '</span>' : '<span class="tag">On this computer</span>';
    const via = m.via === 'model' ? '<span class="tag">chosen by Gemma</span>' : '';
    $('toolcard').innerHTML = `<div class="src-head"><span class="src-title">${TOOL_ICON[m.name] || '🛠'} ${esc(m.title)}</span><span class="src-page">${m.ok ? '' : 'did not work · '}${m.ms != null ? Math.round(m.ms) + ' ms' : ''}</span></div>`
      + `<p class="src-text">${rows || esc(f.error || '')}</p><div class="src-more">${badge} ${via}</div>`;
    $('toolcard').classList.remove('hidden');
    if (m.online && m.ok) { const c = $('net-chip'); c.classList.add('flash'); setTimeout(() => c.classList.remove('flash'), 2500); }
    if (m.name === 'timer' && m.ok && m.args && (m.args.action || 'set') === 'set') {
      timerEnd = Date.now() + ((+m.args.minutes || 0) * 60 + (+m.args.seconds || 0)) * 1000; timerLabel = m.args.label || ''; renderTimer();
    }
    if (m.name === 'timer' && m.ok && m.args && m.args.action === 'cancel') { timerEnd = 0; renderTimer(); }
  }
  function renderTimer() {
    const c = $('timer-chip'); const left = timerEnd - Date.now();
    if (!timerEnd || left <= 0) { c.classList.add('hidden'); return; }
    const s = Math.ceil(left / 1000); c.textContent = `⏱ ${timerLabel ? timerLabel + ' ' : ''}${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    c.classList.remove('hidden');
  }
  setInterval(renderTimer, 500);
  function onDevice(m) {
    if (m.action === 'louder' && outGain) outGain.gain.value = Math.min(2, outGain.gain.value * 1.4);
    else if (m.action === 'quieter' && outGain) outGain.gain.value = Math.max(0.15, outGain.gain.value / 1.4);
    else if (m.action === 'mute' && !muted) toggleMute();
    else if (m.action === 'unmute' && muted) toggleMute();
    else if (m.action === 'voice' && m.voice && voices.some((v) => v.id === m.voice)) { voice = m.voice; setUrl(); renderTop(); }
    toast({ louder: 'Volume up', quieter: 'Volume down', mute: 'Mic muted', unmute: 'Mic on', slower: 'Speaking slower', faster: 'Speaking faster', normal_speed: 'Normal speed', voice: 'Voice changed' }[m.action] || 'Done');
  }

  function showSources(items) {
    const box = $('source'); if (!items.length) { box.classList.add('hidden'); return; }
    const top = items[0];
    const seen = new Set([`${top.title}|${top.page}`]);
    const more = items.slice(1).filter((x) => !seen.has(`${x.title}|${x.page}`) && seen.add(`${x.title}|${x.page}`)).map((x) => `${esc(x.title)} ${esc(pageLabel(x.page))}`).join(' · ');
    box.innerHTML = `<div class="src-head"><span class="src-title">${esc(top.title)}</span><span class="src-page">${esc(pageLabel(top.page))}${top.section && !String(top.page || '').startsWith(top.section) ? ' · ' + esc(top.section) : ''}</span></div>`
      + `<p class="src-text">${esc((top.text || '').slice(0, 320))}${(top.text || '').length > 320 ? '…' : ''}</p>`
      + (more ? `<div class="src-more">Also read: ${more}</div>` : '');
    box.classList.remove('hidden');
  }
  async function loadLibrary() {
    try {
      const { data } = await (await fetch(`/api/library?profile=${profile}`)).json();
      $('lib-title').textContent = 'Library';
      $('lib-list').innerHTML = data.docs.length ? data.docs.map((d) => `<li><span>${esc(d.title)}</span><span class="meta">${esc(d.kind || '')}${d.pages ? ' · ' + Number(d.pages) + ' pp' : ''}${d.chunks ? ' · ' + Number(d.chunks) + ' passages' : ''}${d.licence ? ' · ' + esc(d.licence) : ''}</span></li>`).join('')
        : '<li><span>No documents yet. Add your own with rag_ingest.py.</span></li>';
      $('lib-file').textContent = `${data.docs.length} documents · ${data.file || ''} on this computer`;
    } catch (_) { $('lib-file').textContent = 'Library unavailable'; }
  }

  async function loadHistory() {
    try {
      const { data } = await (await fetch(`/api/sessions?profile=${profile}`)).json();
      $('his-title').textContent = 'Conversations';
      $('his-file').textContent = `${data.sessions.length} sessions · ${data.file} on this computer`;
      const ul = $('his-list'); ul.innerHTML = data.sessions.length ? '' : '<li><span>No conversations yet.</span></li>';
      for (const s of data.sessions) {
        const li = document.createElement('li'); li.className = 'sess';
        const when = new Date(s.started * 1000);
        li.innerHTML = `<button class="sess-btn"><span class="sess-when">${when.toLocaleDateString()} ${when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>`
          + `<span class="sess-first">${esc(s.first || '')}</span><span class="meta">${Number(s.turns)} turns${s.avg_ttfa_ms ? ' · typical reply in ' + (s.avg_ttfa_ms / 1000).toFixed(2) + ' s' : ''}</span></button><button class="del" aria-label="Delete this conversation">×</button><div class="sess-turns hidden"></div>`;
        li.querySelector('.del').onclick = async () => { await fetch(`/api/sessions/${s.id}?profile=${profile}`, { method: 'DELETE' }); loadHistory(); toast('Conversation deleted'); };
        li.querySelector('.sess-btn').onclick = async () => {
          const box = li.querySelector('.sess-turns');
          if (box.classList.toggle('hidden')) return;
          const t = (await (await fetch(`/api/sessions/${s.id}?profile=${profile}`)).json()).data.turns;
          box.innerHTML = t.map((x) => `<div class="tq">${esc(x.user)}</div><div class="ta">${x.filler ? '<i>' + esc(x.filler) + '</i> ' : ''}${esc((x.reply || '').replace(x.filler || '\u0000', '').trim())}${x.interrupted ? ' <em>(interrupted)</em>' : ''}</div>`
            + `<div class="tm">${x.sources.length ? 'From ' + esc(x.sources.join(' · ')) + ' · ' : ''}${x.memories.length ? x.memories.length + ' memories · ' : ''}${x.ttfa_ms ? (x.ttfa_ms / 1000).toFixed(2) + ' s' : ''}</div>`).join('');
        };
        ul.appendChild(li);
      }
    } catch (_) { $('his-file').textContent = 'History unavailable'; }
  }
  $('his-clear').onclick = async () => {
    if (!confirm('Delete all saved conversations?')) return;
    await fetch(`/api/sessions?profile=${profile}`, { method: 'DELETE' }); loadHistory(); toast('History cleared');
  };

  function clearCall() {
    $('you').textContent = ''; $('said').textContent = ''; $('chips').innerHTML = ''; $('source').classList.add('hidden');
    replySentences = []; recalled = new Set(); learned = new Set(); firstAudioT = null; speechEnd = null; resetSteps();
    $('lat').textContent = 'Say something to begin.';
  }

  // ---------------- memory chips + drawer ----------------
  function showRecalled(facts) {
    recalled = new Set(facts.map((f) => f.id));
    $('chips').innerHTML = facts.slice(0, 3).map((f) => `<span class="mchip"><i>Remembered</i>${esc(nice(f.text))}</span>`).join('');
    loadMemory();
  }
  function showLearned(facts) {
    learned = new Set(facts);
    $('chips').insertAdjacentHTML('beforeend', facts.map((f) => `<span class="mchip learn"><i>Learned</i>${esc(nice(f))}</span>`).join(''));
    loadMemory();
  }
  async function loadMemory() {
    try {
      const { data } = await (await fetch(`/api/memory?profile=${profile}`)).json();
      $('mem-title').textContent = 'What Nova knows about you';
      $('mem-file').textContent = `${data.facts.length} memories · stored in ${data.file} on this computer`;
      $('mlist').innerHTML = '';
      for (const f of data.facts) {
        const li = document.createElement('li');
        li.className = recalled.has(f.id) ? 'hit' : (learned.has(f.text) ? 'new' : '');
        li.innerHTML = `<span>${esc(nice(f.text))}</span><span class="meta">${f.source === 'seed' ? '' : 'learned'}${f.hits ? (f.source === 'seed' ? '' : ' · ') + 'recalled ' + f.hits + '×' : ''}</span><button class="del" aria-label="Forget this">×</button>`;
        li.querySelector('.del').onclick = async () => { await fetch(`/api/memory/${f.id}?profile=${profile}`, { method: 'DELETE' }); loadMemory(); toast('Forgotten'); };
        $('mlist').appendChild(li);
      }
    } catch (_) { $('mem-file').textContent = 'Memory unavailable'; }
  }
  async function resetMemory() {
    if (!confirm('Forget everything Nova has learned about you?')) return;
    await fetch(`/api/memory/reset?profile=${profile}`, { method: 'POST' }); recalled = new Set(); learned = new Set(); loadMemory(); toast('Memory cleared');
  }
  $('mem-reset').onclick = resetMemory;

  // ---------------- top bar, drawers, state ----------------
  function renderTop() {
    $('who').textContent = 'Settings';
    const ct = templates.find((t) => t.id === tmpl); $('tmpl-chip').textContent = ct ? ct.name : ''; $('tmpl-chip').classList.toggle('hidden', !ct);
    document.body.dataset.tmpl = tmpl || '';
    const me = profiles.find((x) => x.id === profile); $('kid-chip').classList.toggle('hidden', !(me && me.child));
    const v = voices.find((x) => x.id === voice); $('voice-chip').textContent = v ? `${(v.display || v.id).split(' (')[0]} voice` : 'Voice';
    $('mic-chip').textContent = muted ? 'Mic muted' : 'Mic on';
    $('dev-chip').textContent = 'CPU';
  }
  $('who').onclick = backToStart;
  $('mic-chip').onclick = toggleMute;
  function toggleMute() { muted = !muted; renderTop(); toast(muted ? 'Mic muted' : 'Mic on'); }
  function toggle(drawer, btn, force) {
    const open = force ?? !$(drawer).classList.contains('open');
    $(drawer).classList.toggle('open', open); $(btn).setAttribute('aria-expanded', String(open));
    if (drawer === 'memory' && open) loadMemory();
    if (drawer === 'library' && open) loadLibrary();
    if (drawer === 'history' && open) loadHistory();
  }
  $('mem-btn').onclick = () => toggle('memory', 'mem-btn'); $('mem-close').onclick = () => toggle('memory', 'mem-btn', false);
  $('det-btn').onclick = () => toggle('details', 'det-btn');
  $('lib-btn').onclick = () => toggle('library', 'lib-btn'); $('lib-close').onclick = () => toggle('library', 'lib-btn', false);
  $('his-btn').onclick = () => toggle('history', 'his-btn'); $('his-close').onclick = () => toggle('history', 'his-btn', false);
  $('help-btn').onclick = () => $('help').classList.toggle('hidden');
  const LABEL = { idle: 'Ready', listening: 'Listening', processing: 'Thinking', speaking: 'Speaking', interrupted: 'Listening' };
  function setState(s) { uiState = s; $('state').textContent = LABEL[s] || s; $('orb').classList.toggle('thinking', s === 'processing'); }
  function backToStart() {
    connId++; if (ws) { try { ws.onclose = null; ws.close(); } catch (_) {} } ws = null; stopPlayback();
    toggle('memory', 'mem-btn', false); toggle('details', 'det-btn', false); toggle('library', 'lib-btn', false); toggle('history', 'his-btn', false);
    $('call').classList.add('hidden'); $('start').classList.remove('hidden'); loadStart();
  }
  // ---------------- details: pipeline + turns ----------------
  const STAGES = [['vad', 'Voice activity', 'Silero v6 + GTCRN'], ['turn', 'Turn end', 'Smart Turn v3.2'], ['stt', 'Speech to text', 'Moonshine v2'], ['mem', 'Memory', 'MiniLM + exact scan'], ['rag', 'Library', 'FTS5 + vectors'], ['tool', 'Tools', '15 tools · rules + Gemma calls'], ['llm', 'Language model', 'Gemma 4 E2B'], ['tts', 'Voice', 'Piper (sherpa-onnx)']];
  function renderPipe(tr) {
    const d = pipeInfo || {}; const v = {};
    if (d.vad) v.vad = `${d.vad.soft_ms}/${d.vad.check_ms}/${d.vad.hard_ms} ms`;
    if (d.llm) v.llm = d.llm.runtime || '';
    if (d.tts) v.tts = `${d.tts.voice || ''} ${d.tts.sample_rate ? Math.round(d.tts.sample_rate / 1000) + ' kHz' : ''}`;
    if (tr) {
      v.tool = tr.tool ? `${tr.tool.replace(/_/g, ' ')}${tr.tool_online ? ' · internet' : ''} · ${r0(tr.tool_ms)} ms` : 'not used';
      v.rag = tr.rag_ms != null ? `${tr.rag_gate ? (tr.rag_sources || []).length + ' passages' : 'not used'} · ${tr.rag_ms.toFixed(1)} ms` : 'no library';
      v.turn = `${tr.commit_trigger || ''}`; v.stt = `${r0(tr.stt_ms)} ms`; v.mem = `${tr.memory_hits ?? 0} facts · ${tr.memory_ms != null ? tr.memory_ms.toFixed(1) : '–'} ms`;
      v.llm = `first word ${r0(tr.llm_ttft_ms)} ms`; v.tts = `first sentence ${r0(tr.tts_first_ms)} ms`; v.vad = `wait ${r0(tr.hangover_ms)} ms`; v.denoise = 'on · VAD only';
    }
    $('pipe').innerHTML = STAGES.map(([k, n, m], i) => `<div class="st ${tr ? 'done' : ''}" data-k="${k}"><div class="n">${i + 1} · ${n}</div><div class="m">${m}</div><div class="v">${esc(v[k] || '')}</div></div>`).join('');
  }
  function onTrace(tr) {
    turnNo += 1;
    const heard = (firstAudioT !== null && speechEnd !== null) ? firstAudioT - speechEnd : tr.ttfa_ms;
    if (heard != null) $('lat').innerHTML = `Replied <b>${fmtS(heard)}</b> after you stopped · everything on the CPU`;
    renderPipe(tr);
    const segs = [['a', tr.hangover_ms, 'wait'], ['b', tr.stt_after_commit_ms, 'stt'], ['c', tr.llm_first_sentence_ms, 'gemma'], ['d', tr.tts_first_ms, 'voice']];
    const tot = segs.reduce((a, [, x]) => a + (x || 0), 0) || 1;
    $('tl').innerHTML = segs.map(([c, x, n]) => `<div class="${c}" style="width:${(100 * (x || 0) / tot).toFixed(1)}%">${(x || 0) > tot * 0.12 ? n + ' ' + r0(x) : ''}</div>`).join('');
    const row = document.createElement('tr');
    row.innerHTML = `<td>${turnNo}</td><td>${esc(tr.commit_trigger || '')}</td><td>${r0(tr.hangover_ms)}</td><td>${r0(tr.stt_after_commit_ms)}</td><td>${tr.memory_ms != null ? tr.memory_ms.toFixed(1) : '–'}</td><td>${r0(tr.llm_ttft_ms)}</td><td>${r0(tr.llm_first_sentence_ms)}</td><td>${r0(tr.tts_first_ms)}</td><td><b>${heard != null ? r0(heard) + ' ms' : '–'}</b></td>`;
    $('turns').prepend(row);
  }

  // ---------------- keyboard ----------------
  document.addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input, textarea, select')) return;
    if ($('call').classList.contains('hidden')) { if (e.key === 'Enter') begin(); return; }
    const k = e.key.toLowerCase();
    if (e.key === ' ') { e.preventDefault(); toggleMute(); }
    else if (k === 'l') toggle('library', 'lib-btn');
    else if (k === 'h') toggle('history', 'his-btn');
    else if (k === 'm') toggle('memory', 'mem-btn');
    else if (k === 'd') toggle('details', 'det-btn');
    else if (k === 'r') resetMemory();
    else if (k === '?') $('help').classList.toggle('hidden');
    else if (e.key === 'Escape') {
      const open = ['memory', 'library', 'history', 'details'].filter((d) => $(d).classList.contains('open'));
      if (!$('help').classList.contains('hidden')) $('help').classList.add('hidden');
      else if (open.length) open.forEach((d) => toggle(d, { memory: 'mem-btn', library: 'lib-btn', history: 'his-btn', details: 'det-btn' }[d], false));
      else backToStart();
    }
  });

  // ---------------- orb: driven by the real mic and the real playback level ----------------
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const core = document.querySelector('#orb .core'), halo = document.querySelector('#orb .halo');
  const td = new Uint8Array(512); let outLevel = 0, scale = 1;
  function frame(t) {
    let target = 1, glow = 0.55;
    if (analyser) { analyser.getByteTimeDomainData(td); let pk = 0; for (let i = 0; i < td.length; i++) pk = Math.max(pk, Math.abs(td[i] - 128) / 128); outLevel = outLevel * 0.75 + pk * 0.25; }
    if (uiState === 'speaking') { target = 1 + Math.min(0.35, outLevel * 0.6); glow = 0.7 + outLevel; }
    else if (uiState === 'listening' || uiState === 'interrupted') { target = 1 + Math.min(0.3, (muted ? 0 : micLevel) * 1.6); glow = 0.55 + micLevel; }
    else if (uiState === 'processing') { target = 1.02 + 0.03 * Math.sin(t / 180); glow = 0.65; }
    else { target = 1 + 0.035 * Math.sin(t / 640); }
    scale += (target - scale) * 0.25;
    if (reduce) { core.style.opacity = String(0.75 + Math.min(0.25, (scale - 1) * 2)); }
    else { core.style.transform = `scale(${scale.toFixed(3)})`; halo.style.transform = `scale(${(1.25 + (scale - 1) * 1.6).toFixed(3)})`; }
    halo.style.opacity = String(Math.min(1, glow).toFixed(2));
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  if (qs.get('autostart') === '1') begin();
})();
