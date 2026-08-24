/* NetGuard — client logic. No framework: this is a small enough surface
   that vanilla DOM + fetch keeps it legible and dependency-free. */

const STEP_LABELS = [
  "Reading file", "Selecting features", "Training models", "Scanning held-out flows", "Scoring results",
];

const state = {
  phase: "idle",         // idle | running | done | error
  connPage: 1,
  connFilter: "all",
  connSearch: "",
  logLen: 0,
};

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => Array.from(el.querySelectorAll(sel));

function fmt(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-US");
}
function pct(n, digits = 1) {
  if (n === null || n === undefined) return "—";
  return `${Number(n).toFixed(digits)}%`;
}

/* ---------------------------------------------------------------------- */
/* Upload / dropzone                                                       */
/* ---------------------------------------------------------------------- */

function initDropzone() {
  const zone = $("#dropzone");
  const input = $("#file-input");
  const nameEl = $("#dz-filename");
  const btn = $("#scan-btn");
  let file = null;

  const setFile = (f) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) {
      nameEl.textContent = "Only .csv files are supported.";
      nameEl.style.color = "var(--bad)";
      return;
    }
    file = f;
    nameEl.textContent = f.name;
    nameEl.style.color = "var(--focus)";
    btn.disabled = false;
  };

  zone.addEventListener("click", () => input.click());
  input.addEventListener("change", (e) => setFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("drag"); })
  );
  zone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  btn.addEventListener("click", async () => {
    if (!file) return;
    const topN = Number($("#top-n").value) || 15;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("top_n", topN);
    btn.disabled = true;
    btn.textContent = "Starting…";
    const res = await fetch("/scan", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "Upload failed." }));
      nameEl.textContent = err.error || "Upload failed.";
      nameEl.style.color = "var(--bad)";
      btn.disabled = false;
      btn.textContent = "Run scan";
      return;
    }
    state.phase = "running";
    state.logLen = 0;
    render();
    pollProgress();
  });
}

/* ---------------------------------------------------------------------- */
/* Progress polling                                                        */
/* ---------------------------------------------------------------------- */

let pollTimer = null;

async function pollProgress() {
  clearTimeout(pollTimer);
  const res = await fetch("/progress");
  const data = await res.json();

  if (data.phase === "running") {
    renderRunning(data);
    pollTimer = setTimeout(pollProgress, 700);
  } else if (data.phase === "done") {
    state.phase = "done";
    state.result = data.result;
    state.connPage = 1;
    state.connFilter = "all";
    state.connSearch = "";
    render();
    renderResult(data.result);
    loadConnections();
  } else if (data.phase === "error") {
    state.phase = "error";
    state.error = data;
    render();
  }
}

function renderRunning(data) {
  const view = $("#view-running");
  if (view.classList.contains("hidden")) {
    $$(".view").forEach((v) => v.classList.add("hidden"));
    view.classList.remove("hidden");
  }
  const steps = $$(".run-step");
  steps.forEach((s, i) => {
    s.classList.toggle("done", i < data.step - 1 || (i === data.step - 1 && data.step === 5 && data.phase !== "running"));
    s.classList.toggle("active", i === data.step - 1);
  });

  const log = $("#run-log");
  const lines = data.log || [];
  if (lines.length !== state.logLen) {
    log.innerHTML = lines.map((l, i) =>
      `<div class="line${i >= state.logLen ? " new" : ""}">${escapeHtml(l)}</div>`
    ).join("") + `<span class="cursor"></span>`;
    state.logLen = lines.length;
    log.scrollTop = log.scrollHeight;
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

/* ---------------------------------------------------------------------- */
/* Result render                                                           */
/* ---------------------------------------------------------------------- */

function renderResult(r) {
  $$(".view").forEach((v) => v.classList.add("hidden"));
  $("#view-result").classList.remove("hidden");

  $("#case-id").textContent = caseId();
  $("#case-time").textContent = new Date().toLocaleString("en-US", {
    month: "short", day: "numeric", year: "numeric", hour: "2-digit", minute: "2-digit",
  });

  // Trust line
  const trust = $("#trust-line");
  if (r.held_out) {
    trust.classList.remove("warn");
    trust.innerHTML = `<span class="check">✓</span> Held-out evaluation — every flow below was scored by a model that never saw it during training.`;
  } else {
    trust.classList.add("warn");
    trust.innerHTML = `<span class="check">!</span> No held-out split was available for this file — figures below reflect training data and may read optimistic.`;
  }

  // Stat strip — count up
  const stats = [
    { sel: "#stat-total", val: r.total_connections, lbl: null },
    { sel: "#stat-train", val: r.train_rows, lbl: null },
    { sel: "#stat-tested", val: r.test_rows || r.scanned, lbl: null },
    { sel: "#stat-flagged", val: r.flagged, lbl: null },
  ];
  stats.forEach((s) => countUp($(s.sel), s.val || 0));
  $("#stat-flagged").classList.toggle("accent-bad", (r.flagged || 0) > 0);

  // Waveform
  renderWaveform(r.score_hist, r.rl_threshold_pct);

  // Detector performance panel
  const perfPanel = $("#perf-panel");
  if (r.evaluated) {
    perfPanel.innerHTML = metricRows([
      ["Precision", pct(r.precision * 100, 1)],
      ["Recall", pct(r.recall * 100, 1)],
      ["F1", r.f1.toFixed(3)],
      ["AUC", r.auc.toFixed(3)],
      ["False positives", fmt(r.fp), r.fp > 0],
      ["False negatives", fmt(r.fn), r.fn > 0],
    ]) + `<div class="tuning-note"><b>Threshold tuning score:</b> ${(r.tuning_score * 100).toFixed(0)}%
      — a heuristic used only to pick the anomaly cutoff on validation data, tuned separately
      from the metrics above and not itself an accuracy figure.</div>`;
  } else {
    perfPanel.innerHTML = `<div class="tuning-note">No ground-truth labels were found in this file, so precision/recall can't be computed.
      Showing raw detection counts only.</div>`;
  }

  // Feature importance
  const featPanel = $("#feature-panel");
  if (r.top_features && r.top_features.length) {
    const max = r.top_features.length;
    featPanel.innerHTML = r.top_features.map((f, i) => `
      <div class="feature-row">
        <span class="name" title="${f}">${f}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${100 - (i * (70 / max))}%"></span></span>
      </div>`).join("");
  }

  // Charts
  const chartsEl = $("#charts");
  chartsEl.innerHTML = "";
  const chartKeys = r.evaluated
    ? ["cm_chart", "roc_chart", "met_chart"]
    : ["pie_chart"];
  chartKeys.forEach((k) => {
    if (r[k]) {
      const div = document.createElement("div");
      div.className = "chart-frame";
      div.innerHTML = `<img src="${r[k]}" alt="">`;
      chartsEl.appendChild(div);
    }
  });
  if (r.feat_chart) {
    const div = document.createElement("div");
    div.className = "chart-frame";
    div.innerHTML = `<img src="${r.feat_chart}" alt="">`;
    chartsEl.appendChild(div);
  }
}

function metricRows(rows) {
  return rows.map(([k, v, bad]) =>
    `<div class="metric-row"><span class="k">${k}</span><span class="v${bad ? " bad" : ""}">${v}</span></div>`
  ).join("");
}

function countUp(el, target) {
  if (!el) return;
  const start = 0;
  const dur = 600;
  const t0 = performance.now();
  function step(t) {
    const p = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(Math.round(start + (target - start) * eased));
    if (p < 1) requestAnimationFrame(step);
    else el.textContent = fmt(target);
  }
  requestAnimationFrame(step);
}

function caseId() {
  if (!state._caseId) {
    state._caseId = Math.random().toString(16).slice(2, 8);
  }
  return state._caseId;
}

/* ---------------------------------------------------------------------- */
/* Waveform: real score histogram -> twin SVG traces                       */
/* ---------------------------------------------------------------------- */

function renderWaveform(hist, thresholdPct) {
  const holder = $("#waveform-holder");
  if (!hist || (!hist.normal.some(Boolean) && !hist.anomaly.some(Boolean))) {
    holder.innerHTML = `<div class="waveform-empty"><div class="flatline"></div></div>`;
    return;
  }
  const W = 1000, H = 92, PAD = 4;
  const n = hist.normal.length;
  // Each trace is normalized to its own peak, not a shared one. Attacks are
  // almost always the minority class — sharing a max would flatten the
  // "flagged" trace into invisibility whenever normal traffic dominates,
  // which defeats the point of a twin-trace comparison.
  const maxNormal = Math.max(...hist.normal, 1);
  const maxAnomaly = Math.max(...hist.anomaly, 1);

  const pathFor = (arr, maxV) => {
    const step = W / (n - 1);
    let d = `M 0 ${H - PAD}`;
    arr.forEach((v, i) => {
      const x = i * step;
      const y = H - PAD - (v / maxV) * (H - PAD * 2);
      d += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
    });
    d += ` L ${W} ${H - PAD}`;
    return d;
  };
  const lineFor = (arr, maxV) => {
    const step = W / (n - 1);
    return arr.map((v, i) => {
      const x = i * step;
      const y = H - PAD - (v / maxV) * (H - PAD * 2);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  };

  const threshX = thresholdPct != null ? (thresholdPct / 100) * W : null;

  holder.innerHTML = `
    <svg class="waveform-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line class="baseline" x1="0" y1="${H - PAD}" x2="${W}" y2="${H - PAD}" />
      ${threshX !== null ? `<line class="threshold" x1="${threshX}" y1="0" x2="${threshX}" y2="${H}" />` : ""}
      <path class="fill-ok" d="${pathFor(hist.normal, maxNormal)} Z" />
      <path class="fill-bad" d="${pathFor(hist.anomaly, maxAnomaly)} Z" />
      <path class="trace-ok waveform-draw" d="${lineFor(hist.normal, maxNormal)}" />
      <path class="trace-bad waveform-draw" d="${lineFor(hist.anomaly, maxAnomaly)}" />
    </svg>`;
}

/* ---------------------------------------------------------------------- */
/* Connections ledger                                                       */
/* ---------------------------------------------------------------------- */

async function loadConnections() {
  const params = new URLSearchParams({
    page: state.connPage, page_size: 50, filter: state.connFilter, search: state.connSearch,
  });
  const res = await fetch(`/connections?${params}`);
  const data = await res.json();
  renderConnections(data);
}

function renderConnections(data) {
  const tbody = $("#ledger-body");
  const hasLabels = data.has_labels;
  $$("#ledger-head th.actual-col").forEach((th) => th.style.display = hasLabels ? "" : "none");

  tbody.innerHTML = data.rows.map((r, i) => {
    const isBad = r.verdict === "Suspicious";
    const confClass = r.row_accuracy_pct >= 85 ? "ok" : r.row_accuracy_pct >= 60 ? "mid" : "bad";
    const agreeClass = r.agreement.includes("Anomaly") ? "hi-bad" : r.agreement.includes("Normal") ? "hi-ok" : "";
    return `
      <tr class="reveal" style="animation-delay:${Math.min(i, 12) * 18}ms">
        <td class="idx">${String(r.idx).padStart(4, "0")}</td>
        <td class="verdict ${isBad ? "bad" : "ok"}">${r.verdict}</td>
        <td>${r.attack_type}</td>
        <td class="conf-cell">
          <span class="conf-track"><span class="conf-fill ${confClass}" style="width:${r.row_accuracy_pct}%"></span></span>
          ${r.row_accuracy_pct.toFixed(0)}%
        </td>
        <td class="agree ${agreeClass}">${r.agreement}</td>
        <td>${r.iso_pred}</td>
        <td>${r.rf_pred}</td>
        ${hasLabels ? `<td class="actual-col">${r.true_label ?? "—"}</td>` : ""}
      </tr>`;
  }).join("");

  const from = data.total === 0 ? 0 : (data.page - 1) * data.page_size + 1;
  const to = Math.min(data.page * data.page_size, data.total);
  $("#ledger-count").textContent = `Showing ${fmt(from)}–${fmt(to)} of ${fmt(data.total)}`;
  $("#prev-page").disabled = data.page <= 1;
  $("#next-page").disabled = to >= data.total;
}

function initLedgerControls() {
  $$(".chip[data-filter]").forEach((chip) => {
    chip.addEventListener("click", () => {
      $$(".chip[data-filter]").forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      state.connFilter = chip.dataset.filter;
      state.connPage = 1;
      loadConnections();
    });
  });
  let searchDebounce;
  $("#ledger-search").addEventListener("input", (e) => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => {
      state.connSearch = e.target.value;
      state.connPage = 1;
      loadConnections();
    }, 250);
  });
  $("#prev-page").addEventListener("click", () => {
    if (state.connPage > 1) { state.connPage--; loadConnections(); }
  });
  $("#next-page").addEventListener("click", () => {
    state.connPage++; loadConnections();
  });
}

/* ---------------------------------------------------------------------- */
/* Live capture                                                            */
/* ---------------------------------------------------------------------- */

function initCapture(modelsReady) {
  const head = $("#capture-head");
  const body = $("#capture-body");
  const startBtn = $("#capture-start");
  const stopBtn = $("#capture-stop");
  const feed = $("#capture-feed");
  let open = false;

  head.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    open = !open;
    body.classList.toggle("open", open);
  });

  if (!modelsReady) {
    startBtn.disabled = true;
    $("#capture-hint").textContent = "Run a CSV scan first — live capture needs trained models.";
  }

  startBtn.addEventListener("click", async () => {
    const res = await fetch("/start_capture", { method: "POST" });
    const data = await res.json();
    if (data.error) { $("#capture-hint").textContent = data.error; return; }
    startBtn.disabled = true;
    stopBtn.disabled = false;
    open = true;
    body.classList.add("open");
  });
  stopBtn.addEventListener("click", async () => {
    await fetch("/stop_capture", { method: "POST" });
    startBtn.disabled = false;
    stopBtn.disabled = true;
  });

  if (window.io) {
    const socket = io();
    socket.on("live_flow", (f) => {
      const empty = $(".capture-empty", feed);
      if (empty) empty.remove();
      const row = document.createElement("div");
      row.className = "flow";
      const bad = f.verdict !== "NORMAL";
      row.innerHTML = `
        <span>${f.src}</span><span>${f.dst}:${f.dport}</span>
        <span>${f.proto}</span><span class="${bad ? "v-bad" : "v-ok"}">${f.verdict}</span>
        <span>${f.agreement}</span><span>${f.pkts}</span>`;
      feed.prepend(row);
      while (feed.children.length > 60) feed.removeChild(feed.lastChild);
    });
    socket.on("capture_error", (d) => { $("#capture-hint").textContent = d.msg; });
  }
}

/* ---------------------------------------------------------------------- */
/* View switching + boot                                                   */
/* ---------------------------------------------------------------------- */

function render() {
  $$(".view").forEach((v) => v.classList.add("hidden"));
  if (state.phase === "idle") $("#view-idle").classList.remove("hidden");
  if (state.phase === "running") $("#view-running").classList.remove("hidden");
  if (state.phase === "error") {
    $("#view-error").classList.remove("hidden");
    $("#error-msg").textContent = state.error.error || "Something went wrong.";
    $("#error-trace").textContent = state.error.step_msg || "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initDropzone();
  initLedgerControls();
  const modelsReady = document.body.dataset.modelsReady === "true";
  initCapture(modelsReady);

  $("#new-scan-btn")?.addEventListener("click", () => {
    state.phase = "idle";
    render();
  });
  $("#retry-btn")?.addEventListener("click", () => {
    state.phase = "idle";
    render();
  });
  $("#download-btn")?.addEventListener("click", () => {
    window.location = "/download";
  });

  // Always check server state on load — a scan may already be running or finished
  // (e.g. the page was reloaded, or started from another tab), regardless of whether
  // models from a *previous* scan exist yet.
  fetch("/progress").then((r) => r.json()).then((data) => {
    if (data.phase === "running") {
      state.phase = "running";
      state.logLen = 0;
      render();
      pollProgress();
    } else if (data.phase === "done" && data.result) {
      state.phase = "done";
      renderResult(data.result);
      loadConnections();
    } else if (data.phase === "error") {
      state.phase = "error";
      state.error = data;
      render();
    } else {
      state.phase = "idle";
      render();
    }
  }).catch(() => {
    state.phase = "idle";
    render();
  });
});
