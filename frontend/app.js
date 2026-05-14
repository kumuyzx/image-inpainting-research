const form = document.querySelector("#inpaint-form");
const statusEl = document.querySelector("#status");
const imageInput = document.querySelector("#image-input");
const maskInput = document.querySelector("#mask-input");
const imagePreview = document.querySelector("#image-preview");
const maskPreview = document.querySelector("#mask-preview");
const resultImage = document.querySelector("#result-image");
const resultStage = document.querySelector("#result-stage");
const downloadLink = document.querySelector("#download-link");
const runButton = document.querySelector(".run-button");
const modeValue = document.querySelector("#mode-value");
const elapsedValue = document.querySelector("#elapsed-value");
const cpuValue = document.querySelector("#cpu-value");
const gpuValue = document.querySelector("#gpu-value");

let startedAt = 0;
let elapsedTimer = null;
let statusTimer = null;

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) {
    return "--";
  }
  return `${seconds.toFixed(seconds >= 10 ? 1 : 2)}s`;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}%` : "--";
}

function updateMetrics(payload = {}) {
  const device = payload.device || payload.requestedDevice;
  modeValue.textContent = device ? device.toUpperCase() : "--";
  cpuValue.textContent = formatPercent(payload.cpuPercent);

  if (payload.gpu) {
    const used = payload.gpu.memoryUsedMiB;
    const total = payload.gpu.memoryTotalMiB;
    gpuValue.textContent = `${payload.gpu.utilizationPercent}% / ${used}MB`;
    gpuValue.title = `${used}MB / ${total}MB`;
  } else {
    gpuValue.textContent = device === "cpu" ? "未使用" : "--";
    gpuValue.removeAttribute("title");
  }

  if (payload.activeJob && Number.isFinite(payload.activeJob.elapsedSeconds)) {
    elapsedValue.textContent = formatSeconds(payload.activeJob.elapsedSeconds);
  } else if (payload.lastJob && Number.isFinite(payload.lastJob.elapsedSeconds)) {
    elapsedValue.textContent = formatSeconds(payload.lastJob.elapsedSeconds);
  }
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    updateMetrics(await response.json());
  } catch (_) {
    // The backend may still be loading or already stopped.
  }
}

function startRuntimeTracking() {
  startedAt = performance.now();
  elapsedValue.textContent = "0.00s";
  elapsedTimer = window.setInterval(() => {
    elapsedValue.textContent = formatSeconds((performance.now() - startedAt) / 1000);
  }, 200);
  statusTimer = window.setInterval(refreshStatus, 1000);
  refreshStatus();
}

function stopRuntimeTracking(finalElapsedSeconds) {
  window.clearInterval(elapsedTimer);
  window.clearInterval(statusTimer);
  elapsedTimer = null;
  statusTimer = null;

  if (Number.isFinite(finalElapsedSeconds)) {
    elapsedValue.textContent = formatSeconds(finalElapsedSeconds);
  }
  refreshStatus();
}

function previewFile(input, image) {
  const file = input.files && input.files[0];
  if (!file) {
    image.removeAttribute("src");
    return;
  }
  image.src = URL.createObjectURL(file);
}

imageInput.addEventListener("change", () => previewFile(imageInput, imagePreview));
maskInput.addEventListener("change", () => previewFile(maskInput, maskPreview));

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!imageInput.files[0] || !maskInput.files[0]) {
    setStatus("缺少文件", true);
    return;
  }

  const body = new FormData();
  body.append("image", imageInput.files[0]);
  body.append("mask", maskInput.files[0]);

  runButton.disabled = true;
  setStatus("处理中");
  resultImage.hidden = true;
  downloadLink.hidden = true;
  resultStage.querySelector("span").textContent = "正在修复...";
  startRuntimeTracking();

  try {
    const response = await fetch("/api/inpaint", {
      method: "POST",
      body,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "请求失败");
    }

    const resultUrl = `${payload.resultUrl}?t=${Date.now()}`;
    resultImage.src = resultUrl;
    resultImage.hidden = false;
    resultStage.querySelector("span").textContent = "";
    downloadLink.href = resultUrl;
    downloadLink.hidden = false;
    updateMetrics(payload.status || payload);
    setStatus("完成");
  } catch (error) {
    resultStage.querySelector("span").textContent = error.message;
    setStatus("失败", true);
  } finally {
    stopRuntimeTracking();
    runButton.disabled = false;
  }
});

refreshStatus();
