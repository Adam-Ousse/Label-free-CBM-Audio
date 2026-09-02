const elements = {
  classSelect: document.getElementById("classSelect"),
  status: document.getElementById("classStatus"),
  grid: document.getElementById("examplesGrid"),
  pagination: document.getElementById("pagination"),
  previous: document.getElementById("previousPage"),
  next: document.getElementById("nextPage"),
  pageStatus: document.getElementById("pageStatus"),
  viewIntro: document.getElementById("viewIntro"),
  allCount: document.getElementById("allCount"),
  correctCount: document.getElementById("correctCount"),
  incorrectCount: document.getElementById("incorrectCount"),
};

const state = {
  data: null,
  examples: [],
  view: "intervention",
  outcome: "incorrect",
  classLabel: "all",
  page: 0,
};

const STORAGE_KEY = "audio-lf-cbm-explorer-state-v1";
let renderTimer = null;

const lineColors = ["#176b87", "#dc8b2c", "#287a5c", "#c64c3f", "#8a5ca8"];

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function displayName(value) {
  return String(value || "").replace(/[_-]+/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function percentage(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function signed(value) {
  const number = Number(value) || 0;
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function pageSize() {
  return state.view === "segmented" ? 1 : 4;
}

function saveViewState() {
  try {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
      view: state.view,
      outcome: state.outcome,
      classLabel: state.classLabel,
      page: state.page,
    }));
  } catch (_) {
    // Storage can be unavailable for local file previews or privacy-restricted browsers.
  }
}

function restoreViewState() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (!saved) return;
    if (["intervention", "segmented"].includes(saved.view)) state.view = saved.view;
    if (["all", "correct", "incorrect"].includes(saved.outcome)) state.outcome = saved.outcome;
    if (typeof saved.classLabel === "string") state.classLabel = saved.classLabel;
    if (Number.isInteger(saved.page) && saved.page >= 0) state.page = saved.page;
  } catch (_) {
    // Ignore malformed or inaccessible session state.
  }
}

function scheduleRender({ scroll = false } = {}) {
  if (renderTimer !== null) window.clearTimeout(renderTimer);
  elements.grid.setAttribute("aria-busy", "true");
  renderTimer = window.setTimeout(() => {
    renderTimer = null;
    try {
      render();
      if (scroll) elements.grid.scrollIntoView({ block: "start" });
    } catch (error) {
      elements.status.textContent = `Unable to render this explanation: ${error.message || "unknown error"}`;
      elements.status.classList.add("error");
    } finally {
      elements.grid.removeAttribute("aria-busy");
    }
  }, 0);
}

function filteredExamples() {
  return state.examples.filter((item) => {
    const classMatch = state.classLabel === "all" || item.classLabel === state.classLabel;
    const outcomeMatch = state.outcome === "all" ||
      (state.outcome === "correct" && item.example.explanation.correct) ||
      (state.outcome === "incorrect" && !item.example.explanation.correct);
    return classMatch && outcomeMatch;
  });
}

function cardHeader(item) {
  const ex = item.example;
  const explanation = ex.explanation;
  const outcome = explanation.correct ? "correct" : "incorrect";
  return `
    <div class="card-heading">
      <div>
        <h3>${escapeHtml(item.emoji)} ${escapeHtml(displayName(explanation.gt_class))}</h3>
        <p class="sample-id">${escapeHtml(ex.id)} · ${escapeHtml(ex.selection_reason)}</p>
      </div>
      <span class="outcome-badge ${outcome}">${outcome}</span>
    </div>
    <div class="gt-line"><strong>Ground truth:</strong> ${escapeHtml(displayName(explanation.gt_class))}</div>
    <audio controls preload="none" src="${escapeHtml(ex.audio)}">Audio playback is not supported by this browser.</audio>
  `;
}

function probabilityRows(probabilities) {
  const ranked = probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 3);
  return ranked.map((item) => {
    const label = state.data.classes[item.index].label;
    return `
      <div class="prob-row">
        <span class="prob-label" title="${escapeHtml(displayName(label))}">${escapeHtml(displayName(label))}</span>
        <span class="prob-track"><span class="prob-fill" style="width:${Math.max(1, item.probability * 100).toFixed(2)}%"></span></span>
        <span class="prob-value">${percentage(item.probability)}</span>
      </div>`;
  }).join("");
}

function interventionCard(item) {
  const ex = item.example;
  const explanation = ex.explanation;
  const outcome = explanation.correct ? "correct" : "incorrect";
  const shownConcepts = explanation.top_concepts.slice(0, 5);
  const maxSignal = Math.max(...shownConcepts.map((concept) => Math.abs(concept.contribution)), 1e-8);
  const controls = shownConcepts.map((concept, index) => {
    const relativeSignal = Math.abs(concept.contribution) / maxSignal;
    return `
    <div class="concept-control">
      <span class="concept-name" title="${escapeHtml(concept.concept)}">${escapeHtml(concept.concept)}</span>
      <button type="button" role="slider" class="signal-bar" data-signal-bar data-concept-position="${index}"
        data-factor="1" data-relative-signal="${relativeSignal}" aria-valuemin="0" aria-valuemax="2"
        aria-valuenow="1" aria-label="Change the importance of ${escapeHtml(concept.concept)}">
        <span class="signal-fill ${concept.contribution >= 0 ? "positive" : "negative"}"
          data-signal-fill style="width:${relativeSignal * 50}%"></span>
        <span class="signal-baseline" style="left:${relativeSignal * 50}%" aria-hidden="true"></span>
      </button>
      <span class="concept-value" data-factor-position="${index}">${signed(concept.contribution)} · 1.00×</span>
    </div>
  `;
  }).join("");

  return `
    <article class="example-card ${outcome}" data-example-key="${escapeHtml(item.key)}">
      ${cardHeader(item)}
      <div class="prediction-strip">
        <div class="pred-block">
          <small>Original prediction</small>
          <strong>${escapeHtml(displayName(explanation.pred_class))} · ${percentage(explanation.confidence)}</strong>
        </div>
        <span class="arrow" aria-hidden="true">→</span>
        <div class="pred-block edited">
          <small>After intervention</small>
          <strong data-edited-prediction>${escapeHtml(displayName(explanation.pred_class))} · ${percentage(explanation.confidence)}</strong>
        </div>
      </div>
      <div class="probabilities" data-probabilities></div>
      <div class="intervention-head">
        <h4>Top five concept signals</h4>
        <button type="button" class="reset-button" data-reset>Reset</button>
      </div>
      <div class="concept-controls">${controls}</div>
      <div class="card-meta">
        <span>Fold ${ex.fold}</span><span>${ex.duration_sec.toFixed(1)} s</span><span>Click a bar to intervene</span>
      </div>
    </article>
  `;
}

function bindInterventions(items) {
  items.forEach((item) => {
    const card = elements.grid.querySelector(`[data-example-key="${item.key}"]`);
    if (!card) return;
    const explanation = item.example.explanation;
    const concepts = explanation.top_concepts.slice(0, 5);
    const bars = [...card.querySelectorAll("[data-signal-bar]")];
    const output = card.querySelector("[data-edited-prediction]");
    const probabilityBox = card.querySelector("[data-probabilities]");

    const update = () => {
      const logits = explanation.base_logits.slice();
      bars.forEach((bar) => {
        const position = Number(bar.dataset.conceptPosition);
        const factor = Number(bar.dataset.factor);
        const concept = concepts[position];
        for (let classIndex = 0; classIndex < logits.length; classIndex += 1) {
          logits[classIndex] += concept.class_effects[classIndex] * (factor - 1);
        }
        const relativeSignal = Number(bar.dataset.relativeSignal);
        bar.querySelector("[data-signal-fill]").style.width = `${relativeSignal * factor * 50}%`;
        bar.setAttribute("aria-valuenow", factor.toFixed(2));
        card.querySelector(`[data-factor-position="${position}"]`).textContent =
          `${signed(concept.contribution * factor)} · ${factor.toFixed(2)}×`;
      });
      const probabilities = softmax(logits);
      const prediction = argmax(probabilities);
      const label = state.data.classes[prediction].label;
      output.textContent = `${displayName(label)} · ${percentage(probabilities[prediction])}`;
      output.classList.toggle("changed", prediction !== explanation.pred_index);
      probabilityBox.innerHTML = probabilityRows(probabilities);
    };

    const setFactor = (bar, value) => {
      const factor = Math.max(0, Math.min(2, Math.round(value * 20) / 20));
      bar.dataset.factor = String(factor);
      update();
    };

    bars.forEach((bar) => {
      bar.addEventListener("click", (event) => {
        const bounds = bar.getBoundingClientRect();
        const relativeSignal = Math.max(Number(bar.dataset.relativeSignal), 1e-8);
        const targetWidth = (event.clientX - bounds.left) / bounds.width;
        setFactor(bar, (targetWidth * 2) / relativeSignal);
      });
      bar.addEventListener("keydown", (event) => {
        let factor = Number(bar.dataset.factor);
        if (event.key === "ArrowLeft" || event.key === "ArrowDown") factor -= 0.05;
        else if (event.key === "ArrowRight" || event.key === "ArrowUp") factor += 0.05;
        else if (event.key === "Home") factor = 0;
        else if (event.key === "End") factor = 2;
        else return;
        event.preventDefault();
        setFactor(bar, factor);
      });
    });
    card.querySelector("[data-reset]").addEventListener("click", () => {
      bars.forEach((bar) => { bar.dataset.factor = "1"; });
      update();
    });
    update();
  });
}

function lineChart(temporal) {
  const width = 620;
  const height = 255;
  const margin = { left: 48, right: 12, top: 12, bottom: 32 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allValues = temporal.series.flatMap((series) => series.values).concat([0]);
  let low = Math.min(...allValues);
  let high = Math.max(...allValues);
  if (Math.abs(high - low) < 1e-8) { low -= 1; high += 1; }
  const pad = (high - low) * 0.08;
  low -= pad;
  high += pad;
  const x = (index) => margin.left + (index / Math.max(1, temporal.centers_sec.length - 1)) * plotWidth;
  const y = (value) => margin.top + ((high - value) / (high - low)) * plotHeight;

  const horizontal = Array.from({ length: 5 }, (_, index) => {
    const value = high - (index / 4) * (high - low);
    return `<line class="grid" x1="${margin.left}" x2="${width - margin.right}" y1="${y(value)}" y2="${y(value)}"></line>
      <text x="${margin.left - 7}" y="${y(value) + 3}" text-anchor="end">${value.toFixed(2)}</text>`;
  }).join("");
  const timeLabels = temporal.centers_sec.map((time, index) =>
    `<text x="${x(index)}" y="${height - 10}" text-anchor="middle">${Number(time).toFixed(1)}</text>`
  ).join("");
  const lines = temporal.series.map((series, seriesIndex) => {
    const points = series.values.map((value, index) => `${x(index)},${y(value)}`).join(" ");
    const circles = series.values.map((value, index) =>
      `<circle cx="${x(index)}" cy="${y(value)}" r="2.4" fill="${lineColors[seriesIndex]}"></circle>`
    ).join("");
    return `<polyline points="${points}" fill="none" stroke="${lineColors[seriesIndex]}" stroke-width="2.1" stroke-linejoin="round"></polyline>${circles}`;
  }).join("");
  const legend = temporal.series.map((series, index) =>
    `<span title="${escapeHtml(series.concept)}"><i style="background:${lineColors[index]}"></i>${escapeHtml(displayName(series.concept))}</span>`
  ).join("");

  return `<svg class="line-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Temporal contribution line chart">
      ${horizontal}
      <line class="zero" x1="${margin.left}" x2="${width - margin.right}" y1="${y(0)}" y2="${y(0)}"></line>
      ${timeLabels}${lines}
      <text x="${width / 2}" y="${height}" text-anchor="middle">Time (s)</text>
    </svg><div class="chart-legend">${legend}</div>`;
}

function heatmap(temporal) {
  const maxMagnitude = Math.max(
    ...temporal.local.flatMap((segment) => segment.map((item) => Math.abs(item.contribution))),
    1e-8,
  );
  const header = temporal.centers_sec.map((time) => `<th>${Number(time).toFixed(1)} s</th>`).join("");
  const rows = Array.from({ length: 5 }, (_, rank) => {
    const cells = temporal.local.map((segment) => {
      const item = segment[rank];
      const ratio = Math.abs(item.contribution) / maxMagnitude;
      const positive = item.contribution >= 0;
      const color = positive ? `rgba(23,107,135,${0.12 + ratio * 0.78})` : `rgba(198,76,63,${0.12 + ratio * 0.78})`;
      const textColor = ratio > 0.56 ? "#ffffff" : "#17232d";
      return `<td style="background:${color};color:${textColor}" title="${escapeHtml(item.concept)} · ${signed(item.contribution)}">${escapeHtml(displayName(item.concept))}</td>`;
    }).join("");
    return `<tr><th>Rank ${rank + 1}</th>${cells}</tr>`;
  }).join("");
  return `<div class="heatmap-wrap"><table class="heatmap" aria-label="Top concepts in each audio segment"><thead><tr><th></th>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function segmentedCard(item) {
  const ex = item.example;
  const explanation = ex.explanation;
  const outcome = explanation.correct ? "correct" : "incorrect";
  return `
    <article class="example-card ${outcome}">
      ${cardHeader(item)}
      <div class="prediction-strip">
        <div class="pred-block"><small>Full-clip prediction</small><strong>${escapeHtml(displayName(explanation.pred_class))} · ${percentage(explanation.confidence)}</strong></div>
        <span class="arrow" aria-hidden="true">·</span>
        <div class="pred-block"><small>Segmentation</small><strong>${state.data.window_sec.toFixed(1)} s window · ${state.data.hop_sec.toFixed(1)} s hop</strong></div>
      </div>
      <div class="segment-plots">
        <section class="plot-panel">
          <h4>(a) Aggregate top-5 concept evolution</h4>
          <p class="plot-subtitle">Signed contribution to the original full-clip predicted class</p>
          ${lineChart(ex.temporal)}
        </section>
        <section class="plot-panel">
          <h4>(b) Per-segment top-5 concepts</h4>
          <p class="plot-subtitle">Color intensity encodes contribution magnitude; hover for the exact value</p>
          ${heatmap(ex.temporal)}
        </section>
      </div>
    </article>`;
}

function render() {
  elements.status.classList.remove("error");
  const matches = filteredExamples();
  const itemsPerPage = pageSize();
  const pages = Math.max(1, Math.ceil(matches.length / itemsPerPage));
  state.page = Math.max(0, Math.min(state.page, pages - 1));
  const start = state.page * itemsPerPage;
  const items = matches.slice(start, start + itemsPerPage);

  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === state.view;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll("[data-outcome]").forEach((button) => {
    button.classList.toggle("active", button.dataset.outcome === state.outcome);
  });

  elements.viewIntro.innerHTML = state.view === "intervention"
    ? "Click inside a concept bar to set its signal between <strong>0×</strong> and <strong>2×</strong>. The thin marker is the original 1× signal. The site recomputes all 50 class logits from the saved sparse classifier effects while holding the other concepts fixed."
    : "The left plot follows five concepts through time; the right plot shows the five strongest local contributors in each window. Contributions target the original full-clip predicted class.";

  if (!items.length) {
    elements.grid.innerHTML = `<div class="empty">No ${escapeHtml(state.outcome)} showcase example is available for this class. Try another outcome or choose all classes.</div>`;
    elements.status.textContent = "No matching examples.";
  } else {
    elements.grid.innerHTML = items.map((item) => state.view === "intervention" ? interventionCard(item) : segmentedCard(item)).join("");
    if (state.view === "intervention") bindInterventions(items);
    elements.status.textContent = `Showing ${start + 1}–${start + items.length} of ${matches.length} curated ${state.outcome === "all" ? "test" : state.outcome} examples.`;
  }

  elements.pagination.classList.toggle("hidden", matches.length <= itemsPerPage);
  elements.previous.disabled = state.page === 0;
  elements.next.disabled = state.page >= pages - 1;
  elements.pageStatus.textContent = `Page ${state.page + 1} of ${pages}`;
  saveViewState();
}

function initialize(data) {
  if (!data || Number(data.schema_version) < 2 || !Array.isArray(data.classes)) {
    throw new Error("The showcase manifest is missing LF+broad intervention data. Rebuild the docs assets.");
  }
  state.data = data;
  state.examples = data.classes.flatMap((cls) => cls.examples.map((example) => ({
    key: `sample-${example.id}`,
    classLabel: cls.label,
    emoji: cls.emoji || "🔊",
    example,
  })));

  data.classes.forEach((cls) => {
    const option = document.createElement("option");
    option.value = cls.label;
    option.textContent = `${cls.emoji || "🔊"} ${displayName(cls.label)}`;
    elements.classSelect.appendChild(option);
  });
  restoreViewState();
  if (state.classLabel !== "all" && !data.classes.some((cls) => cls.label === state.classLabel)) {
    state.classLabel = "all";
    state.page = 0;
  }
  elements.classSelect.value = state.classLabel;

  const correct = state.examples.filter((item) => item.example.explanation.correct).length;
  const incorrect = state.examples.length - correct;
  elements.allCount.textContent = `(${state.examples.length})`;
  elements.correctCount.textContent = `(${correct})`;
  elements.incorrectCount.textContent = `(${incorrect})`;

  elements.classSelect.addEventListener("change", () => {
    state.classLabel = elements.classSelect.value;
    state.page = 0;
    saveViewState();
    scheduleRender();
  });
  document.querySelectorAll("[data-outcome]").forEach((button) => button.addEventListener("click", () => {
    state.outcome = button.dataset.outcome;
    state.page = 0;
    saveViewState();
    scheduleRender();
  }));
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    if (state.view === tab.dataset.view) return;
    state.view = tab.dataset.view;
    state.page = 0;
    saveViewState();
    scheduleRender();
  }));
  elements.previous.addEventListener("click", () => { state.page -= 1; scheduleRender({ scroll: true }); });
  elements.next.addEventListener("click", () => { state.page += 1; scheduleRender({ scroll: true }); });
  render();
}

async function load() {
  try {
    if (window.ESC50_SHOWCASE) {
      initialize(window.ESC50_SHOWCASE);
      return;
    }
    const response = await fetch("assets/data/esc50_showcase.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
    initialize(await response.json());
  } catch (error) {
    elements.status.textContent = error.message || "Unable to load showcase data.";
    elements.status.classList.add("error");
  }
}

load();
