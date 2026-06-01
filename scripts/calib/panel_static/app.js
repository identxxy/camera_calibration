const state = {
  modes: {},
  selectedMode: "",
  requestedMode: new URLSearchParams(window.location.search).get("mode") || "",
  requestedModeApplied: false,
  jobs: [],
  selectedJobId: "",
  logOffset: 0,
};

const els = {
  serverStatus: document.getElementById("server-status"),
  modeList: document.getElementById("mode-list"),
  modeFields: document.getElementById("mode-fields"),
  modeForm: document.getElementById("mode-form"),
  dryRun: document.getElementById("dry-run"),
  startJob: document.getElementById("start-job"),
  refreshJobs: document.getElementById("refresh-jobs"),
  jobCount: document.getElementById("job-count"),
  jobList: document.getElementById("job-list"),
  jobDetail: document.getElementById("job-detail"),
  jobLog: document.getElementById("job-log"),
  cancelJob: document.getElementById("cancel-job"),
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {"Content-Type": "application/json"},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function statusClass(status) {
  return `status ${status || ""}`;
}

function setStatus(text) {
  els.serverStatus.textContent = text;
}

function renderModes() {
  const entries = Object.entries(state.modes);
  els.modeList.innerHTML = entries.map(([key, mode]) => `
    <button class="mode-button ${key === state.selectedMode ? "active" : ""}" data-mode="${key}" type="button">
      <strong>${escapeHtml(mode.title)}</strong>
      <span>${escapeHtml(mode.operator_summary || key)}</span>
    </button>
  `).join("");
  els.modeList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedMode = button.dataset.mode;
      updateModeUrl(state.selectedMode);
      renderModes();
      renderForm();
    });
  });
}

function renderForm() {
  const mode = state.modes[state.selectedMode];
  if (!mode) {
    els.modeFields.innerHTML = "";
    return;
  }
  els.modeFields.innerHTML = mode.params.map((field) => {
    const value = field.default ?? "";
    if (field.type === "checkbox") {
      return `
        <label class="checkbox-field">
          <input name="${escapeAttr(field.name)}" type="checkbox" ${value ? "checked" : ""}>
          <span>${escapeHtml(field.label)}</span>
        </label>
      `;
    }
    return `
      <div class="field">
        <label for="field-${escapeAttr(field.name)}">${escapeHtml(field.label)}</label>
        <input
          id="field-${escapeAttr(field.name)}"
          name="${escapeAttr(field.name)}"
          type="${field.type === "number" ? "number" : "text"}"
          value="${escapeAttr(value)}"
          placeholder="${escapeAttr(field.placeholder || "")}">
      </div>
    `;
  }).join("");
}

function collectParams() {
  const mode = state.modes[state.selectedMode];
  const params = {};
  for (const field of mode.params) {
    const input = els.modeForm.elements[field.name];
    if (!input) continue;
    if (field.type === "checkbox") {
      params[field.name] = input.checked;
    } else if (field.type === "number") {
      params[field.name] = input.value === "" ? "" : Number(input.value);
    } else {
      params[field.name] = input.value;
    }
  }
  return params;
}

async function startJob(event) {
  event.preventDefault();
  if (!state.selectedMode) return;
  els.startJob.disabled = true;
  try {
    const params = collectParams();
    const pipelinePreview = state.selectedMode.startsWith("run_")
      && state.selectedMode.endsWith("_recalib_pipeline")
      && params.pipeline_dry_run === true;
    const job = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify({
        mode: state.selectedMode,
        params,
        dry_run: els.dryRun.checked && !pipelinePreview,
      }),
    });
    state.selectedJobId = job.id;
    state.logOffset = 0;
    await refreshJobs();
    await refreshSelectedJob();
  } catch (error) {
    alert(error.message);
  } finally {
    els.startJob.disabled = false;
  }
}

async function refreshModes() {
  const payload = await api("/api/modes");
  state.modes = payload.modes || {};
  if (!state.requestedModeApplied && state.requestedMode && state.modes[state.requestedMode]) {
    state.selectedMode = state.requestedMode;
    state.requestedModeApplied = true;
  } else if (!state.selectedMode) {
    state.selectedMode = Object.keys(state.modes)[0] || "";
  }
  renderModes();
  renderForm();
}

async function refreshJobs() {
  const payload = await api("/api/jobs");
  state.jobs = payload.jobs || [];
  if (!state.selectedJobId && state.jobs.length) {
    state.selectedJobId = state.jobs[0].id;
    state.logOffset = 0;
  }
  renderJobs();
}

function renderJobs() {
  els.jobCount.textContent = String(state.jobs.length);
  els.jobList.innerHTML = state.jobs.map((job) => `
    <button class="job-button ${job.id === state.selectedJobId ? "active" : ""}" data-job-id="${escapeAttr(job.id)}" type="button">
      <strong>${escapeHtml(job.mode_title || job.mode)}</strong>
      <span><span class="${statusClass(job.status)}">${escapeHtml(job.status)}</span> ${escapeHtml(job.created_at || "")}</span>
      <span>${escapeHtml(job.id)}</span>
    </button>
  `).join("");
  els.jobList.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", async () => {
      state.selectedJobId = button.dataset.jobId;
      state.logOffset = 0;
      els.jobLog.textContent = "";
      renderJobs();
      await refreshSelectedJob();
    });
  });
}

async function refreshSelectedJob() {
  if (!state.selectedJobId) {
    els.jobDetail.textContent = "No job selected.";
    els.jobDetail.classList.add("empty");
    els.cancelJob.disabled = true;
    return;
  }
  try {
    const job = await api(`/api/jobs/${encodeURIComponent(state.selectedJobId)}`);
    renderJobDetail(job);
    await refreshLog(job);
  } catch (error) {
    els.jobDetail.textContent = error.message;
    els.jobDetail.classList.add("empty");
  }
}

function renderJobDetail(job) {
  els.jobDetail.classList.remove("empty");
  els.cancelJob.disabled = !["pending", "running"].includes(job.status);
  const reports = (job.reports || []).map((report) => {
    const href = report.exists ? (report.url || report.panel_url) : report.url;
    const link = report.exists
      ? `<a href="${escapeAttr(href)}" target="_blank" rel="noreferrer">${escapeHtml(report.label)}</a>`
      : `<strong>${escapeHtml(report.label)}</strong>`;
    return `
      <div class="report">
        ${link}
        <small>${report.exists ? "ready" : "pending"}: ${escapeHtml(report.path || "")}</small>
      </div>
    `;
  }).join("");

  const steps = (job.steps || []).map((step, index) => `
    <div class="step">
      <strong>${index + 1}. ${escapeHtml(step.name || "")}</strong>
      <span class="${statusClass(step.status)}">${escapeHtml(step.status || "pending")}</span>
      ${step.command ? `<code>${escapeHtml(step.command)}</code>` : ""}
    </div>
  `).join("");

  els.jobDetail.innerHTML = `
    <div class="section-head">
      <h3>${escapeHtml(job.mode_title || job.mode)}</h3>
      <span class="${statusClass(job.status)}">${escapeHtml(job.status)}</span>
    </div>
    <div class="detail-grid">
      <div class="metric"><span>Run dir</span><strong>${escapeHtml(job.run_dir || "")}</strong></div>
      <div class="metric"><span>Started</span><strong>${escapeHtml(job.started_at || "-")}</strong></div>
      <div class="metric"><span>Dry run</span><strong>${job.dry_run ? "yes" : "no"}</strong></div>
    </div>
    ${job.error ? `<div class="step"><strong>Error</strong><code>${escapeHtml(job.error)}</code></div>` : ""}
    <h3>Reports</h3>
    <div class="reports">${reports || "<div class='report'>No report artifacts declared.</div>"}</div>
    <h3>Steps</h3>
    <div class="steps">${steps}</div>
  `;
}

async function refreshLog(job) {
  const chunk = await api(`/api/jobs/${encodeURIComponent(job.id)}/log?offset=${state.logOffset}`);
  if (chunk.text) {
    els.jobLog.textContent += chunk.text;
    els.jobLog.scrollTop = els.jobLog.scrollHeight;
  }
  state.logOffset = chunk.next_offset;
}

async function cancelSelectedJob() {
  if (!state.selectedJobId) return;
  els.cancelJob.disabled = true;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.selectedJobId)}/cancel`, {
      method: "POST",
      body: "{}",
    });
    await refreshJobs();
    await refreshSelectedJob();
  } catch (error) {
    alert(error.message);
  }
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  })[ch]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function updateModeUrl(mode) {
  if (!mode || !window.history || !window.history.replaceState) return;
  const url = new URL(window.location.href);
  url.searchParams.set("mode", mode);
  window.history.replaceState(null, "", url);
}

async function tick() {
  try {
    await refreshJobs();
    await refreshSelectedJob();
    setStatus("Ready");
  } catch (error) {
    setStatus(error.message);
  }
}

els.modeForm.addEventListener("submit", startJob);
els.refreshJobs.addEventListener("click", tick);
els.cancelJob.addEventListener("click", cancelSelectedJob);

(async function init() {
  try {
    await refreshModes();
    await tick();
    setInterval(tick, 2000);
  } catch (error) {
    setStatus(error.message);
  }
})();
