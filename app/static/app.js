const $ = (id) => document.getElementById(id);
const drop = $("drop"), input = $("file"), statusEl = $("status");
let lastResult = null;

fetch("/api/warm", { method: "POST" }).catch(() => {});

function setStatus(html, err = false) {
  statusEl.innerHTML = html;
  statusEl.classList.toggle("err", err);
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// 0 -> green, 0.5 -> amber, 1 -> red
function heatColor(p, alpha = 1) {
  const stops = [[47, 182, 124], [216, 169, 59], [229, 72, 77]];
  const t = Math.min(Math.max(p, 0), 1) * 2;
  const [a, b] = t <= 1 ? [stops[0], stops[1]] : [stops[1], stops[2]];
  const f = t <= 1 ? t : t - 1;
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

// Meter position on a log-odds scale clamped to [0.001, 0.999]; a linear bar
// would squeeze the 0.99-1.0 "likely AI" band into 1% of its width.
function meterPos(p) {
  const L = 7, q = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  const x = Math.min(Math.max(Math.log(q / (1 - q)), -L), L);
  return ((x + L) / (2 * L)) * 100;
}

function fmtPct(p) {
  if (p >= 0.999) return ">99.9%";
  if (p <= 0.001) return "<0.1%";
  return (p * 100).toFixed(p > 0.99 || p < 0.01 ? 1 : 0) + "%";
}

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
});
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
input.addEventListener("change", () => { if (input.files[0]) analyze(input.files[0]); input.value = ""; });
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) analyze(item.getAsFile());
});
$("again").addEventListener("click", () => {
  $("result").hidden = true; drop.classList.remove("compact");
  window.scrollTo({ top: 0 }); input.click();
});
$("heatToggle").addEventListener("change", (e) => { $("heat").style.opacity = e.target.checked ? 1 : 0; });
window.addEventListener("resize", () => lastResult && drawHeat(lastResult));

async function analyze(file) {
  if (file.size > 15 * 1024 * 1024) return setStatus("That file is over 15 MB.", true);
  $("result").hidden = true;
  const slow = setTimeout(() => setStatus(
    '<span class="spinner"></span>Starting a GPU. The first check after a quiet period takes up to a minute…'), 8000);
  setStatus('<span class="spinner"></span>Analyzing…');

  const fd = new FormData();
  fd.append("file", file, file.name || "pasted.png");
  let res, body;
  try {
    res = await fetch("/api/analyze", { method: "POST", body: fd });
    body = await res.json().catch(() => ({}));
  } catch {
    clearTimeout(slow);
    return setStatus("Network error. Please try again.", true);
  }
  clearTimeout(slow);
  if (!res.ok) return setStatus(esc(body.detail || `Error ${res.status}`), true);

  setStatus("");
  const url = URL.createObjectURL(file);
  const img = $("preview");
  img.onload = () => { render(body); URL.revokeObjectURL(url); };
  img.onerror = () => { render(body); };  // e.g. HEIC the browser can't show
  img.src = url;
}

function render(r) {
  lastResult = r;
  $("result").hidden = false;
  drop.classList.add("compact");

  const p = r.likelihood, v = r.verdict.key;
  $("verdict").className = "verdict " + v;
  $("vLabel").textContent = r.verdict.label;
  $("vScore").textContent = fmtPct(p);

  const lo = r.bands.likely_real, hi = r.bands.likely_ai;
  const pl = meterPos(lo), ph = meterPos(hi);
  document.querySelector(".z-real").style.width = pl + "%";
  document.querySelector(".z-mid").style.width = ph - pl + "%";
  document.querySelector(".z-ai").style.width = 100 - ph + "%";
  $("tLow").style.left = pl + "%"; $("tLow").textContent = lo.toFixed(2);
  $("tMid").style.left = meterPos(0.5) + "%";
  $("tHigh").style.left = ph + "%"; $("tHigh").textContent = hi.toFixed(2);
  $("needle").style.left = meterPos(p) + "%";

  const explain = {
    likely_ai: "The classifier is highly confident this image was generated or heavily edited by AI.",
    likely_real: "The classifier found no strong sign of AI generation. That does not prove the image is authentic, because recent generators can evade detection.",
    inconclusive: "The score falls in the range where this model is unreliable. Treat it as unknown, and use the heatmap and metadata below as extra context.",
  }[v];
  $("explain").textContent = explain;

  // Provenance callout
  const m = r.metadata, prov = $("provenance");
  if (m.declared_ai) {
    prov.className = "callout ai"; prov.hidden = false;
    prov.innerHTML = "<strong>The file's own metadata declares it AI-generated.</strong> This is independent of the classifier and is strong evidence.";
  } else if (m.camera) {
    prov.className = "callout camera"; prov.hidden = false;
    prov.innerHTML = `<strong>Camera metadata present:</strong> ${esc(m.camera)}. EXIF can be edited, so treat this as a hint, not proof.`;
  } else prov.hidden = true;

  $("metaList").innerHTML = m.findings.map((f) =>
    `<li><span class="k">${esc(f.label)}${f.signal !== "neutral" ? `<span class="tag ${f.signal}">${f.signal === "ai" ? "AI" : f.signal}</span>` : ""}</span><span class="v">${esc(f.value)}</span></li>`
  ).join("") + `<li><span class="k">File</span><span class="v">${esc(r.file.format)} · ${r.image.width}×${r.image.height} · ${(r.file.bytes / 1024).toFixed(0)} kB</span></li>`;

  // Tile grid
  const tg = $("tileGrid");
  tg.style.gridTemplateColumns = `repeat(${r.grid || 1}, 1fr)`;
  tg.innerHTML = r.tiles.map((t) =>
    `<div style="background:${heatColor(t.score)}" title="row ${t.row + 1}, col ${t.col + 1}">${Math.round(t.score * 100)}</div>`).join("")
    || '<p class="fine">Image too small for region analysis.</p>';

  $("diagList").innerHTML = r.indicators.map((d) =>
    `<li><span class="k">${esc(d.label)}</span><span class="v">${d.score.toFixed(4)}${d.weight ? ' <span class="tag info">scored</span>' : ""}<span class="note">${esc(d.note)}</span></span></li>`).join("");

  $("modelInfo").textContent = `${r.model_version} · ${r.elapsed_s}s`;
  drawHeat(r);
  $("result").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawHeat(r) {
  const img = $("preview"), cv = $("heat");
  const w = img.clientWidth, h = img.clientHeight;
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  for (const t of r.tiles) {
    // Opacity scales with score so clean images stay readable.
    const a = 0.12 + 0.43 * t.score;
    ctx.fillStyle = heatColor(t.score, a);
    ctx.fillRect(t.x * w, t.y * h, t.width * w, t.height * h);
    ctx.strokeStyle = "rgba(0,0,0,.25)";
    ctx.strokeRect(t.x * w, t.y * h, t.width * w, t.height * h);
  }
}
