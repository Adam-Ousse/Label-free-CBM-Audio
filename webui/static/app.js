const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const statusEl = document.getElementById("status");
const fileMetaEl = document.getElementById("fileMeta");
const topKEl = document.getElementById("topK");
const resultPanel = document.getElementById("resultPanel");
const predClassEl = document.getElementById("predClass");
const predConfEl = document.getElementById("predConf");
const modelDirEl = document.getElementById("modelDir");
const influenceFigureEl = document.getElementById("influenceFigure");
const playerEl = document.getElementById("player");

let selectedFile = null;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = "status" + (kind ? ` ${kind}` : "");
}

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(2)} ${units[idx]}`;
}

function setFile(file) {
  selectedFile = file;
  analyzeBtn.disabled = !selectedFile;

  if (!selectedFile) {
    fileMetaEl.classList.add("hidden");
    playerEl.classList.add("hidden");
    playerEl.removeAttribute("src");
    return;
  }

  fileMetaEl.textContent = `Selected: ${file.name} (${formatBytes(file.size)})`;
  fileMetaEl.classList.remove("hidden");

  const audioURL = URL.createObjectURL(file);
  playerEl.src = audioURL;
  playerEl.classList.remove("hidden");

  setStatus("Ready for analysis.");
}

function handleFiles(files) {
  if (!files || files.length === 0) {
    setFile(null);
    return;
  }
  setFile(files[0]);
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => handleFiles(event.target.files));

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    event.stopPropagation();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const files = event.dataTransfer ? event.dataTransfer.files : null;
  handleFiles(files);
});

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) {
    setStatus("Please select a file first.", "error");
    return;
  }

  const topK = Math.max(3, Math.min(20, Number(topKEl.value || 10)));
  topKEl.value = String(topK);

  const formData = new FormData();
  formData.append("audio", selectedFile);
  formData.append("top_k", String(topK));

  analyzeBtn.disabled = true;
  setStatus("Analyzing audio with CBM...", "");

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Analysis failed");
    }

    predClassEl.textContent = payload.predicted_class;
    predConfEl.textContent = `${(payload.confidence * 100).toFixed(2)}%`;
    modelDirEl.textContent = payload.model_dir;
    influenceFigureEl.src = `data:image/png;base64,${payload.plot_base64}`;
    resultPanel.classList.remove("hidden");

    setStatus("Analysis complete.", "success");
  } catch (error) {
    setStatus(error.message || "Unexpected error", "error");
  } finally {
    analyzeBtn.disabled = !selectedFile;
  }
});
