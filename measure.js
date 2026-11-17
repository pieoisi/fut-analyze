import "./debug.js";
import { mountAffectRating, resetAffectRating } from "./affect-rating.js";

const SR = 16000, WIN = 5, PREP = 5, REC = 20, FREE = 60, REPS = 2;

// 読み上げ課題は最小限に絞る（静寂と通常の読み上げのみ）
const LABEL = {
  silence: "静かにして、何も話さない",
  read: "下の英文を、落ち着いて読む",
};

// 読み上げ用の英文プール（毎回ランダムに選択されます）
const TEXT_POOL = [
  "Once upon a time, a kind girl named Cinderella lived with her stepmother and two stepsisters. Every day, she cleaned the house and cooked the meals, but she never complained.",
  "Learning a new language can be challenging, but it is also very rewarding. Consistent practice every day helps you build confidence and fluency over time.",
  "Technology has completely changed the way we communicate with each other. People can now talk to friends and colleagues on the other side of the world in real time.",
  "Breakfast is often considered the most important meal of the day because it gives you the energy you need to start your morning productively."
];

// 自由課題の質問プール（ここから毎回ランダムに複数選ばれます）
const FREE_POOL = [
  "What did you do on your last day off?",
  "Which country would you like to visit, and why?",
  "What is your favorite food, and why do you like it?",
  "What is something you would like to try in the future?",
  "Can you describe your ideal daily routine?",
  "What was the most memorable trip you have ever taken?",
  "How do you usually handle stress or a busy schedule?",
  "What is a skill you would love to learn, and why?"
];

// セッションで出題する自由課題の数
const NUM_FREE_QUESTIONS = 4;

const COLS = ["participant", "device", "ns", "t_ms", "event", "source", "trial", "condition", "rep", "phase", "audio_sample_index", "audio_sample_rate_hz", "audio_saved", "audio_sec", "A", "D", "V", "rms_win", "rms_1s", "act_1s", "infer_ms", "affect_valence", "affect_arousal", "affect_label", "affect_av_t_ms", "affect_av_age_ms", "rating_label", "note"];

const $ = (id) => document.getElementById(id);
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / (a.length || 1));
const IN_APP = /Line\/|FBAN|FBAV|Instagram|MicroMessenger|; wv\)/i.test(navigator.userAgent);
let session, ctx, stream, timer, t0 = 0, busy = false, meta = {};
let rows = [], wavChunks = [], buf = new Float32Array(0);
let latestTrialAV = null;
let audioNode, nextAudioMarkId = 0;
const pendingAudioMarks = new Map();
let cur = { trial: "", cond: "", rep: "", phase: "setup" };

mountAffectRating($("affect-grid"), (rating) => {
  const age = latestTrialAV ? Math.round(performance.now() - t0) - latestTrialAV.affect_av_t_ms : "";
  log("affect_rating", {
    ...rating,
    ...(latestTrialAV ?? {}),
    affect_av_age_ms: age,
  });
  $("affect-rated").textContent = "記録しました。次の課題へ進んでください。";
  updateNextEnabled();
}, { once: false });

const canShareWav = (() => {
  if (typeof File !== "function" || typeof navigator.canShare !== "function") return false;
  try {
    return navigator.canShare({ files: [new File([""], "a.wav", { type: "audio/wav" })] });
  } catch {
    return false;
  }
})();
if (IN_APP && !canShareWav) {
  $("wav").checked = false;
  $("wav").closest("label").hidden = true;
}

function log(event, x = {}) {
  rows.push({ participant: meta.pid, device: meta.dev, ns: meta.ns, t_ms: Math.round(performance.now() - t0),
    event, source: "measure", trial: cur.trial, condition: cur.cond, rep: cur.rep, phase: cur.phase, ...x });
}

function markAudio(event) {
  if (!audioNode) throw new Error("音声処理が開始されていません");
  const id = String(++nextAudioMarkId);
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingAudioMarks.delete(id);
      reject(new Error(`音声区間の記録に失敗しました: ${event}`));
    }, 3000);
    pendingAudioMarks.set(id, { event, resolve, reject, timeout });
    try {
      audioNode.port.postMessage({ type: "mark", id });
    } catch (error) {
      clearTimeout(timeout);
      pendingAudioMarks.delete(id);
      reject(error);
    }
  });
}

function handleAudioMessage({ data }) {
  if (data?.type === "marker") {
    const mark = pendingAudioMarks.get(data.id);
    if (!mark) {
      console.error(`未要求の音声区間マーカーを受信しました: ${data.id}`);
      return;
    }
    clearTimeout(mark.timeout);
    pendingAudioMarks.delete(data.id);
    const audioSampleIndex = data.sampleIndex;
    log(mark.event, {
      audio_sample_index: audioSampleIndex,
      audio_sample_rate_hz: SR,
      audio_sec: audioSampleIndex / SR,
    });
    mark.resolve(audioSampleIndex);
    return;
  }
  if (data instanceof Float32Array) {
    push(data);
    return;
  }
  console.error("音声処理から不明なデータを受信しました", data);
}

function push(c) {
  if (meta.wav) {
    const i = new Int16Array(c.length);
    for (let k = 0; k < c.length; k++) i[k] = Math.max(-1, Math.min(1, c[k])) * 32767;
    wavChunks.push(i);
  }
  const m = new Float32Array(Math.min(buf.length + c.length, SR * 8));
  const keep = m.length - c.length;
  m.set(buf.subarray(buf.length - keep), 0);
  m.set(c, keep);
  buf = m;
}

async function tick() {
  if (busy) return;
  busy = true;
  const phaseAtStart = cur.phase;
  try {
    const last = buf.subarray(Math.max(0, buf.length - SR));
    let act = 0, n = 0;
    for (let i = 0; i + 320 <= last.length; i += 320, n++) if (rms(last.subarray(i, i + 320)) > 0.01) act++;
    const x = { rms_1s: rms(last).toFixed(4), act_1s: n ? (act / n).toFixed(2) : "" };
    if (buf.length >= WIN * SR) {
      const w = buf.slice(buf.length - WIN * SR), t = performance.now();
      const r = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", w, [1, w.length]) });
      const [a, d, v] = r[session.outputNames[0]].data;
      Object.assign(x, { A: a.toFixed(3), D: d.toFixed(3), V: v.toFixed(3),
        rms_win: rms(w).toFixed(4), infer_ms: Math.round(performance.now() - t) });
      if (phaseAtStart === "rec") {
        latestTrialAV = { A: x.A, D: x.D, V: x.V, affect_av_t_ms: Math.round(performance.now() - t0) };
      }
    }
    log("tick", x);
  } finally { busy = false; }
}

function phase(name, sec) {
  cur.phase = name;
  $("phase").textContent = { prep: "準備（この間の値は分析から除外）", rec: "計測中" }[name];
  return new Promise((res) => {
    const end = performance.now() + sec * 1000;
    const iv = setInterval(() => {
      const left = Math.ceil((end - performance.now()) / 1000);
      $("count").textContent = left > 0 ? `${left}秒` : "";
      if (left <= 0) { clearInterval(iv); res(); }
    }, 200);
  });
}

function updateNextEnabled() {
  const hasRate = !$("rate").hidden;
  const hasAffect = !$("affect-rate").hidden;
  const rateSelected = [...document.querySelectorAll("#rating-options button")]
    .some((button) => button.getAttribute("aria-pressed") === "true");
  const affectSelected = [...document.querySelectorAll("#affect-grid button")]
    .some((button) => button.getAttribute("aria-pressed") === "true");
  $("next-trial").disabled = (hasRate && !rateSelected) || (hasAffect && !affectSelected);
}

function waitForNextClick() {
  cur.phase = "await_next";
  const hasRating = !$("rate").hidden || !$("affect-rate").hidden;
  $("phase").textContent = hasRating
    ? "評価を選んで「次へ」を押してください"
    : "準備ができたら「次へ」を押してください";
  $("count").textContent = "";
  const next = $("next-trial");
  updateNextEnabled();
  next.hidden = false;
  return new Promise((resolve) => {
    next.addEventListener("click", () => {
      log("next");
      next.hidden = true;
      resolve();
    }, { once: true });
  });
}

// 配列をランダムにシャッフルするヘルパー関数
function shuffle(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

async function run() {
  const list = [];
  for (const cond of Object.keys(LABEL)) {
    for (let rep = 1; rep <= REPS; rep++) {
      list.push({ cond, rep });
    }
  }
  // 読み上げ課題の順序をランダム化
  const shuffledList = shuffle(list);

  // 自由課題のプールからランダムに指定数を選び、シャッフルまたはそのまま適用
  const shuffledFreePool = shuffle(FREE_POOL);
  const selectedFreeQs = shuffledFreePool.slice(0, Math.min(NUM_FREE_QUESTIONS, shuffledFreePool.length));
  const freeTrials = selectedFreeQs.map((q, i) => ({ cond: "free", rep: i + 1, q }));
  const trials = [...shuffledList, ...freeTrials];
  trials.forEach((trial) => { trial.affectSample = trial.cond !== "silence"; });
  const readTexts = shuffle(TEXT_POOL); // 読み上げの文は、同じ文を2回使わない
  let readIdx = 0;
  const affectTrialNumbers = trials.flatMap((t, i) => t.affectSample ? [i + 1] : []);
  log("order", { note: `${trials.map((t) => t.cond + t.rep).join(" ")} | affect sample trials: ${affectTrialNumbers.join(" ")}` });

  for (let i = 0; i < trials.length; i++) {
    const t = trials[i];
    $("rate").hidden = true;
    $("rated").textContent = "";
    document.querySelectorAll("#rating-options button").forEach((button) => {
      button.setAttribute("aria-pressed", "false");
    });
    $("affect-rate").hidden = true;
    $("affect-rated").textContent = "";

    // 読み上げ課題の場合、毎回ランダムにテキストプールから1つ選ぶ
    let currentText = "", textId = "";
    if (!t.q && t.cond === "read") {
      const idx = TEXT_POOL.indexOf(readTexts[readIdx++ % readTexts.length]);
      currentText = TEXT_POOL[idx];
      textId = `text${idx}`;
    }

    cur = { trial: i + 1, cond: t.cond, rep: t.rep, phase: "prep" };
    $("prog").textContent = `試行 ${i + 1} / ${trials.length}`;
    $("inst").textContent = t.q ? `英語で答えてください: ${t.q}（詰まったら、下のボタンを押す）` : LABEL[t.cond];
    $("text").textContent = t.q || t.cond === "silence" ? "" : currentText;
    $("stuck").hidden = !t.q; // 「いま詰まった」ボタンは自由会話のときだけ
    latestTrialAV = null;

    log("trial_start", { note: t.q ? `q${FREE_POOL.indexOf(t.q)}: ${t.q}` : textId }); // どの文・どの質問かを記録
    await phase("prep", PREP);
    latestTrialAV = null;
    const recSeconds = t.q ? FREE : REC;
    cur.phase = "rec";
    await markAudio("rec_start");
    await phase("rec", recSeconds);
    await markAudio("rec_end");
    $("rate").hidden = t.cond === "silence"; $("rated").textContent = "";
    $("affect-rate").hidden = !t.affectSample;
    $("affect-rated").textContent = "";
    resetAffectRating($("affect-grid"));
    $("stuck").hidden = true;
    await waitForNextClick();
    $("rate").hidden = true;
    $("affect-rate").hidden = true;
  }
  cur.phase = "done";
  $("affect-rated").textContent = "";
  $("rated").textContent = "";
  log("session_end");
  clearInterval(timer);
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  $("run").hidden = true;
  $("done").hidden = false;
  $("dlwav").hidden = !meta.wav;
}

$("start").addEventListener("click", async () => {
  const pid = $("pid").value.trim();
  if (!/^[A-Za-z0-9_-]+$/.test(pid)) { alert("参加者IDは、半角の英数字で入力してください（例: P02）"); return; }
  $("start").disabled = true;
  try {
    meta = { pid, dev: $("dev").value.trim(), ns: $("ns").checked, wav: $("wav").checked };
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";
    session = await ort.InferenceSession.create("models/wav2small.onnx", { executionProviders: ["wasm"] });
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: meta.ns, autoGainControl: true },
    });
    ctx = new AudioContext();
    await ctx.resume();
    await ctx.audioWorklet.addModule("js/audio-processor.js");
    const node = new AudioWorkletNode(ctx, "resampler-processor");
    audioNode = node;
    audioNode.port.onmessage = handleAudioMessage;
    const mute = ctx.createGain();
    mute.gain.value = 0;
    ctx.createMediaStreamSource(stream).connect(node).connect(mute).connect(ctx.destination);
    try { await navigator.wakeLock?.request("screen"); } catch {}
    t0 = performance.now();
    log("session_start", {
      audio_sample_rate_hz: SR,
      audio_saved: meta.wav,
      note: `${navigator.userAgent} | sampleRate=${ctx.sampleRate} | mic=${$("mic").value} | win=${WIN}s | tool=v3.1`,
    });
    timer = setInterval(tick, 1000);
    $("setup").hidden = true;
    $("run").hidden = false;
    run().catch((error) => {
      console.error(error);
      cur.phase = "error";
      log("session_error", { note: error.message });
      clearInterval(timer);
      stream.getTracks().forEach((track) => track.stop());
      ctx.close();
      $("run").hidden = true;
      $("done").hidden = false;
      $("done-message").textContent = "計測が中断されました。途中までのCSV/WAVを保存できます。";
      $("dlwav").hidden = !meta.wav;
      alert(`計測を続けられませんでした: ${error.message}`);
    });
  } catch (e) {
    console.error(e);
    alert(`開始できませんでした: ${e.message}`);
    $("start").disabled = false;
  }
});

document.querySelectorAll("#rating-options button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll("#rating-options button").forEach((button) => {
    button.setAttribute("aria-pressed", String(button === b));
  });
  log("rating", { note: b.dataset.v ?? "", rating_label: b.dataset.label });
  $("rated").textContent = b.dataset.v
    ? "記録しました。別の項目に変更できます。"
    : "回答なしとして記録しました。別の項目に変更できます。";
  updateNextEnabled();
}));

$("stuck").addEventListener("click", () => { log("stuck_btn"); navigator.vibrate?.(30); });

async function save(name, blob) {
  const file = typeof File === "function" ? new File([blob], name, { type: blob.type }) : null;
  let canShareFile = false;
  if (file && navigator.share && navigator.canShare) {
    try {
      canShareFile = navigator.canShare({ files: [file] });
    } catch {}
  }
  if (canShareFile) {
    try {
      await navigator.share({ files: [file], title: name });
      return "shared";
    } catch (e) {
      if (e.name === "AbortError") return "cancel";
    }
  }
  if (IN_APP) return "unsupported";

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  return "downloaded";
}
const stamp = () => {
  const d = new Date(), p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};

function buildCsv() {
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  return [COLS.join(","), ...rows.map((r) => COLS.map((c) => esc(r[c])).join(","))].join("\n");
}

$("dlcsv").addEventListener("click", async () => {
  const csv = buildCsv();
  const result = await save(`${meta.pid}_${stamp()}.csv`, new Blob(["\uFEFF" + csv], { type: "text/csv" }));
  $("csvbox").hidden = result === "shared" || result === "downloaded";
  $("csvcopy").hidden = result === "shared" || result === "downloaded";
  $("savemsg").textContent = {
    shared: "共有シートを開きました。「ファイルに保存」などを選んでください。",
    cancel: "キャンセルされました。もう一度押すか、下のコピー欄を使ってください。",
    downloaded: "保存しました。",
    unsupported: "このブラウザでは保存できません。下のコピー欄を使ってください。",
  }[result];
  if (result !== "shared" && result !== "downloaded") $("csvbox").value = csv;
});

$("csvcopy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("csvbox").value);
    $("savemsg").textContent = "コピーしました。トークなどに貼り付けて送ってください。";
  } catch {
    $("csvbox").select();
    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch {}
    $("savemsg").textContent = copied
      ? "コピーしました。トークなどに貼り付けて送ってください。"
      : "コピーできませんでした。欄を選択してコピーしてください。";
  }
});

$("dlwav").addEventListener("click", async () => {
  const n = wavChunks.reduce((s, c) => s + c.length, 0);
  const b = new ArrayBuffer(44 + n * 2), v = new DataView(b);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of wavChunks) { new Int16Array(b, o, c.length).set(c); o += c.length * 2; }
  const result = await save(`${meta.pid}_${stamp()}.wav`, new Blob([b], { type: "audio/wav" }));
  $("savemsg").textContent = {
    shared: "共有シートを開きました。",
    cancel: "キャンセルされました。もう一度押してください。",
    downloaded: "保存しました。",
    unsupported: "このブラウザではWAVを保存できません。外部ブラウザで開いてください。",
  }[result];
});
