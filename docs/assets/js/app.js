const classGridEl = document.getElementById("classGrid");
const classStatusEl = document.getElementById("classStatus");
const examplesPanelEl = document.getElementById("examplesPanel");
const examplesTitleEl = document.getElementById("examplesTitle");
const examplesGridEl = document.getElementById("examplesGrid");

let showcaseData = null;
let selectedClassLabel = null;

function setStatus(text, kind = "") {
  classStatusEl.textContent = text;
  classStatusEl.className = `status ${kind}`.trim();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatLabel(label) {
  return label.replace(/_/g, " ");
}

function renderConceptBars(topConcepts) {
  if (!Array.isArray(topConcepts) || topConcepts.length === 0) {
    return `<p class="error-msg">No concept contributions available for this sample.</p>`;
  }

  const maxAbs = Math.max(...topConcepts.map((x) => Math.abs(Number(x.score) || 0)), 1e-6);

  return `
    <div class="concept-plot">
      <div class="concept-legend">
        <span class="legend-item"><span class="legend-swatch pos"></span> Positive</span>
        <span class="legend-item"><span class="legend-swatch neg"></span> Negative</span>
      </div>
      ${topConcepts
        .map((item) => {
          const score = Number(item.score) || 0;
          const concept = escapeHtml(item.concept || "concept");
          const pct = Math.min(100, (Math.abs(score) / maxAbs) * 100);
          const cls = score >= 0 ? "pos" : "neg";
          return `
            <div class="concept-row">
              <div class="concept-label" title="${concept}">${concept}</div>
              <div class="concept-bar ${cls}">
                <div class="concept-fill" style="width: ${pct.toFixed(2)}%;"></div>
              </div>
              <div class="concept-score ${cls}">${score >= 0 ? "+" : ""}${score.toFixed(3)}</div>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderClassGrid(classes) {
  classGridEl.innerHTML = "";

  classes.forEach((cls) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "class-tile";
    btn.dataset.label = cls.label;
    btn.innerHTML = `
      <span class="emoji">${escapeHtml(cls.emoji || "🔊")}</span>
      <span class="name">${escapeHtml(formatLabel(cls.label))}</span>
    `;

    btn.addEventListener("click", () => {
      selectedClassLabel = cls.label;
      document.querySelectorAll(".class-tile").forEach((tile) => {
        tile.classList.toggle("active", tile.dataset.label === selectedClassLabel);
      });
      renderExamples(cls);
    });

    classGridEl.appendChild(btn);
  });
}

function renderExamples(cls) {
  const examples = cls.examples || [];
  examplesTitleEl.textContent = `${formatLabel(cls.label)} ${cls.emoji || ""}`.trim();

  if (examples.length !== 2) {
    examplesGridEl.innerHTML = `<p class="error-msg">Expected exactly 2 examples, found ${examples.length}.</p>`;
    examplesPanelEl.classList.remove("hidden");
    return;
  }

  examplesGridEl.innerHTML = examples
    .map((ex, idx) => {
      const safeId = escapeHtml(ex.id || `sample_${idx + 1}`);
      const safeAudio = escapeHtml(ex.audio || "");
      const safeFold = escapeHtml(String(ex.fold ?? "-"));
      const safeDuration = escapeHtml(String(ex.duration_sec ?? "-"));
      const exp = ex.explanation || {};
      const predClass = escapeHtml(formatLabel(exp.pred_class || "-"));
      const gtClass = escapeHtml(formatLabel(exp.gt_class || "-"));
      const conf = Number(exp.confidence);
      const confText = Number.isFinite(conf) ? `${(conf * 100).toFixed(2)}%` : "-";
      const barsHtml = renderConceptBars(exp.top_concepts || []);
      return `
        <article class="example-card">
          <div class="card-header">
            <h3>Example ${idx + 1}</h3>
            <p>${safeId}</p>
          </div>
          <div class="pred-meta">
            <span><strong>GT:</strong> ${gtClass}</span>
            <span><strong>Pred:</strong> ${predClass}</span>
            <span><strong>Conf:</strong> ${confText}</span>
          </div>
          <audio controls preload="none" src="${safeAudio}"></audio>
          ${barsHtml}
          <div class="meta">
            <span>Fold: ${safeFold}</span>
            <span>Duration: ${safeDuration}s</span>
          </div>
        </article>
      `;
    })
    .join("");

  examplesPanelEl.classList.remove("hidden");
}

function initializeShowcase(data) {
  showcaseData = data || {};
  const classes = showcaseData.classes || [];
  if (!classes.length) {
    throw new Error("Manifest has no classes.");
  }

  renderClassGrid(classes);
  setStatus(`Loaded ${classes.length} classes. Select one to see 2 examples.`, "ok");

  const first = classes[0];
  if (first) {
    selectedClassLabel = first.label;
    const firstButton = classGridEl.querySelector(`.class-tile[data-label="${CSS.escape(first.label)}"]`);
    if (firstButton) {
      firstButton.classList.add("active");
    }
    renderExamples(first);
  }
}

async function loadShowcase() {
  setStatus("Loading ESC-50 showcase data...");

  // file:// pages cannot use fetch reliably due browser CORS policy,
  // so prefer script-injected data when available.
  if (typeof window.ESC50_SHOWCASE !== "undefined") {
    try {
      initializeShowcase(window.ESC50_SHOWCASE);
      return;
    } catch (err) {
      setStatus(err.message || "Invalid embedded showcase data.", "error");
      return;
    }
  }

  try {
    const resp = await fetch("assets/data/esc50_showcase.json", { cache: "no-cache" });
    if (!resp.ok) {
      throw new Error(`Failed to load manifest (${resp.status})`);
    }

    const data = await resp.json();
    initializeShowcase(data);
  } catch (err) {
    const hint = window.location.protocol === "file:" ? " Open via local server, or keep esc50_showcase.js next to index.html." : "";
    setStatus((err.message || "Unable to load showcase.") + hint, "error");
  }
}

loadShowcase();
