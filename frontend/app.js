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

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
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
    setStatus("完成");
  } catch (error) {
    resultStage.querySelector("span").textContent = error.message;
    setStatus("失败", true);
  } finally {
    runButton.disabled = false;
  }
});
