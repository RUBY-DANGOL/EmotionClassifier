// Tab logic
const tabs = document.querySelectorAll(".tab-button");
const tabPanels = { upload: document.getElementById("tab-upload"), camera: document.getElementById("tab-camera") };
tabs.forEach(btn => btn.addEventListener("click", () => {
  tabs.forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  Object.values(tabPanels).forEach(el => el.classList.remove("active"));
  tabPanels[btn.dataset.tab].classList.add("active");
}));

// Upload flow
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const predictUploadBtn = document.getElementById("predict-upload");

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    predictUploadBtn.disabled = false;
  };
  reader.readAsDataURL(file);
});

predictUploadBtn.addEventListener("click", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  setResultsLoading(true);
  try {
    const res = await fetch("/upload", { method: "POST", body: form });
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    renderResults({ error: e.message });
  } finally {
    setResultsLoading(false);
  }
});

// Camera flow
const startBtn = document.getElementById("start-camera");
const captureBtn = document.getElementById("capture");
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

let stream = null;

startBtn.addEventListener("click", async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    captureBtn.disabled = false;
  } catch (e) {
    alert("Camera access failed: " + e.message);
  }
});

captureBtn.addEventListener("click", async () => {
  const w = video.videoWidth;
  const h = video.videoHeight;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, w, h);
  const dataUrl = canvas.toDataURL("image/png"); // base64
  setResultsLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });
    const data = await res.json();
    renderResults(data);
  } catch (e) {
    renderResults({ error: e.message });
  } finally {
    setResultsLoading(false);
  }
});

// Results rendering
const resultsBody = document.getElementById("results-body");

function setResultsLoading(isLoading){
  if (isLoading) {
    resultsBody.innerHTML = '<div class="subtle">Predicting…</div>';
  }
}

function renderResults(data){
  if (data.error) {
    resultsBody.innerHTML = `<div class="warning">Error: ${data.error}</div>`;
    return;
  }
  if (!data.predictions || data.predictions.length === 0){
    resultsBody.innerHTML = `<div class="note">No predictions returned from the model.</div>`;
    return;
  }
  resultsBody.innerHTML = "";
  data.predictions.forEach(item => {
    const row = document.createElement("div");
    row.className = "result-row";
    const label = document.createElement("div");
    label.textContent = `${item.label} (${(item.confidence*100).toFixed(2)}%)`;
    const barWrap = document.createElement("div");
    barWrap.style.flex = "1";
    barWrap.style.marginLeft = "12px";
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${Math.max(2, Math.min(100, item.confidence*100))}%`;
    barWrap.appendChild(bar);
    row.appendChild(label);
    row.appendChild(barWrap);
    resultsBody.appendChild(row);
  });
}