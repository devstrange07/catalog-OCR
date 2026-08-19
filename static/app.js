(() => {
  const form = document.getElementById("uploadForm");
  const fileInput = document.getElementById("fileInput");
  const urlInput = document.getElementById("urlInput");
  const dropzone = document.getElementById("dropzone");
  const fileName = document.getElementById("fileName");
  const submitBtn = document.getElementById("submitBtn");
  const pipelineSection = document.getElementById("pipelineSection");
  const resultsSection = document.getElementById("resultsSection");
  const jobMeta = document.getElementById("jobMeta");
  const errorBox = document.getElementById("errorBox");
  const productBody = document.getElementById("productBody");
  const counts = document.getElementById("counts");
  const parsedText = document.getElementById("parsedText");
  const artifacts = document.getElementById("artifacts");
  const progressBar = document.getElementById("progressBar");
  const progressPct = document.getElementById("progressPct");
  const progressCurrent = document.getElementById("progressCurrent");
  const progressLog = document.getElementById("progressLog");
  const parseStageHint = document.getElementById("parseStageHint");

  let selectedFile = null;
  let pollTimer = null;
  let lastLogLen = 0;

  function urlValue() {
    return (urlInput?.value || "").trim();
  }

  function selectedParseChannel() {
    const checked = form.querySelector('input[name="parseChannel"]:checked');
    return checked?.value === "gemini" ? "gemini" : "paddleocr";
  }

  function channelLabel(channel) {
    return channel === "gemini" ? "Gemini Vision" : "PaddleOCR";
  }

  function setParseStageHint(channel) {
    if (!parseStageHint) return;
    parseStageHint.textContent =
      channel === "gemini"
        ? "Gemini Vision → structure only"
        : "PaddleOCR + Docling → structure only";
  }

  function updateSubmitEnabled() {
    submitBtn.disabled = !(selectedFile || urlValue());
  }

  function setFile(file) {
    selectedFile = file || null;
    fileName.textContent = selectedFile ? selectedFile.name : "";
    if (selectedFile && urlInput) {
      urlInput.value = "";
    }
    updateSubmitEnabled();
  }

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files?.[0];
    if (file) {
      fileInput.files = e.dataTransfer.files;
      setFile(file);
    }
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files?.[0]));
  urlInput?.addEventListener("input", () => {
    if (urlValue() && selectedFile) {
      selectedFile = null;
      fileInput.value = "";
      fileName.textContent = "";
    }
    updateSubmitEnabled();
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const url = urlValue();
    if (!selectedFile && !url) return;

    submitBtn.disabled = true;
    submitBtn.textContent = selectedFile ? "Uploading…" : "Fetching URL…";
    errorBox.hidden = true;
    resultsSection.hidden = true;
    pipelineSection.hidden = false;
    resetStages(Boolean(url && !selectedFile));
    resetProgress();
    const parseChannel = selectedParseChannel();
    setParseStageHint(parseChannel);

    try {
      let job;
      if (selectedFile) {
        const body = new FormData();
        body.append("file", selectedFile);
        body.append("parse_channel", parseChannel);
        const res = await fetch("/api/jobs", { method: "POST", body });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `Upload failed (${res.status})`);
        }
        job = await res.json();
      } else {
        const res = await fetch("/api/jobs/from-url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url, parse_channel: parseChannel }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          const detail = err.detail;
          throw new Error(
            typeof detail === "string" ? detail : `URL job failed (${res.status})`
          );
        }
        job = await res.json();
      }
      jobMeta.textContent = `Job ${job.id} · ${job.filename} · ${channelLabel(job.meta?.parse_channel || parseChannel)}`;
      submitBtn.textContent = "Running…";
      syncCrawlStageVisibility(job.stages);
      renderProgress(job);
      startPolling(job.id);
    } catch (err) {
      errorBox.hidden = false;
      errorBox.textContent = err.message || String(err);
      submitBtn.disabled = false;
      submitBtn.textContent = "Run pipeline";
    }
  });

  function syncCrawlStageVisibility(stages) {
    const crawlLi = document.querySelector('#stageList li[data-stage="crawl"]');
    if (!crawlLi) return;
    crawlLi.hidden = !stages?.crawl;
  }

  function resetStages(showCrawl) {
    document.querySelectorAll("#stageList li").forEach((li) => {
      const badge = li.querySelector("[data-badge]");
      badge.textContent = "pending";
      badge.className = "badge";
      if (li.dataset.stage === "crawl") {
        li.hidden = !showCrawl;
      }
    });
  }

  function resetProgress() {
    lastLogLen = 0;
    progressBar.style.width = "0%";
    progressPct.textContent = "0%";
    progressCurrent.textContent = "Waiting to start…";
    progressLog.innerHTML = "";
  }

  function updateStages(stages) {
    syncCrawlStageVisibility(stages);
    Object.entries(stages || {}).forEach(([key, info]) => {
      const li = document.querySelector(`#stageList li[data-stage="${key}"]`);
      if (!li) return;
      const badge = li.querySelector("[data-badge]");
      badge.textContent = info.status;
      badge.className = `badge ${info.status}`;
    });
  }

  function formatTime(iso) {
    if (!iso) return "--:--:--";
    try {
      const d = new Date(iso);
      return d.toLocaleTimeString(undefined, { hour12: false });
    } catch {
      return "--:--:--";
    }
  }

  function renderProgress(job) {
    const p = job.progress;
    const pct = p?.percent != null ? Math.max(0, Math.min(100, p.percent)) : null;
    if (pct != null) {
      progressBar.style.width = `${pct}%`;
      progressPct.textContent = `${pct}%`;
    }
    if (p?.message) {
      progressCurrent.textContent = p.message;
    }

    const log = job.progress_log || [];
    if (log.length !== lastLogLen) {
      const items = [...log].reverse();
      progressLog.innerHTML = items
        .map((ev) => {
          const failed = ev.step === "failed" ? " failed" : "";
          return `<li>
            <span class="log-time">${escapeHtml(formatTime(ev.ts))}</span>
            <span class="log-msg${failed}">[${escapeHtml(ev.stage)}] ${escapeHtml(ev.message)}</span>
          </li>`;
        })
        .join("");
      lastLogLen = log.length;
    }
  }

  function startPolling(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    const tick = async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) throw new Error("Failed to fetch job");
        const job = await res.json();
        updateStages(job.stages);
        renderProgress(job);
        const channel = channelLabel(job.meta?.parse_channel);
        jobMeta.textContent = `Job ${job.id} · ${job.filename} · ${job.status} · ${channel}`;
        setParseStageHint(job.meta?.parse_channel);

        if (job.error) {
          errorBox.hidden = false;
          errorBox.textContent = job.error;
        }

        if (job.status === "completed" || job.status === "failed") {
          clearInterval(pollTimer);
          pollTimer = null;
          submitBtn.disabled = false;
          submitBtn.textContent = "Run pipeline";
          if (job.status === "completed") renderResults(job);
          else if (job.products?.length) renderResults(job);
        }
      } catch (err) {
        clearInterval(pollTimer);
        pollTimer = null;
        errorBox.hidden = false;
        errorBox.textContent = err.message || String(err);
        submitBtn.disabled = false;
        submitBtn.textContent = "Run pipeline";
      }
    };
    tick();
    pollTimer = setInterval(tick, 500);
  }

  function renderResults(job) {
    resultsSection.hidden = false;
    const validated = job.validated_count ?? 0;
    const needsReview = job.needs_review_count ?? 0;
    const invalid = job.invalid_count ?? 0;
    counts.textContent =
      `${validated} validated · ${needsReview} needs review · ${invalid} invalid · ` +
      `${job.candidate_count ?? 0} candidates`;

    productBody.innerHTML = "";
    (job.products || []).forEach((p) => {
      const tr = document.createElement("tr");
      const issueMessages = (p.validation_errors || []).map((e) =>
        typeof e === "string" ? e : e.message
      );
      const notes = [...issueMessages, ...(p.warnings || [])].join("; ");
      const price =
        p.price != null
          ? `${p.currency ? p.currency + " " : ""}${p.price}`
          : "—";
      tr.innerHTML = `
        <td><span class="status-pill ${p.status}">${p.status.replace("_", " ")}</span></td>
        <td>${escapeHtml(p.name || "")}</td>
        <td>${escapeHtml(p.brand || "—")}</td>
        <td>${escapeHtml(p.sku || p.product_code || "—")}</td>
        <td>${escapeHtml(price)}</td>
        <td>${p.page ?? "—"}</td>
        <td>${p.confidence != null ? p.confidence.toFixed(2) : "—"}</td>
        <td class="notes">${escapeHtml(notes || "—")}</td>
      `;
      productBody.appendChild(tr);
    });

    parsedText.textContent = job.parsed_preview || "(no preview)";

    const names = ["parsed.json", "candidates.json", "products.json"];
    if (job.stages?.crawl) names.unshift("crawl.json");
    artifacts.innerHTML = names
      .map(
        (n) =>
          `<a href="/api/jobs/${job.id}/artifacts/${n}" target="_blank" rel="noopener">${n}</a>`
      )
      .join("");
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }
})();
