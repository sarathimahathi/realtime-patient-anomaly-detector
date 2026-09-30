// Real-Time Patient Anomaly Detector Client
(function () {
  const patientId = "PATIENT-1001";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host || "localhost:8000";
  const wsUrl = `${protocol}//${host}/ws/live/${patientId}?interval=1.0`;

  let socket = null;
  const historyPoints = [];
  const maxHistory = 40;

  // DOM Elements
  const connectionStatus = document.getElementById("connectionStatus");
  const connectionText = document.getElementById("connectionText");
  const statusDot = connectionStatus.querySelector(".status-dot");
  const injectBtn = document.getElementById("injectBtn");
  const lastUpdated = document.getElementById("lastUpdated");

  const riskScoreEl = document.getElementById("riskScore");
  const riskFill = document.getElementById("riskFill");
  const severityBadge = document.getElementById("severityBadge");

  const valHeartRate = document.getElementById("valHeartRate");
  const valBP = document.getElementById("valBP");
  const valMAP = document.getElementById("valMAP");
  const valSpO2 = document.getElementById("valSpO2");
  const valRespRate = document.getElementById("valRespRate");
  const valTemp = document.getElementById("valTemp");
  const valShockIndex = document.getElementById("valShockIndex");

  const cardHeartRate = document.getElementById("cardHeartRate");
  const cardBP = document.getElementById("cardBP");
  const cardSpO2 = document.getElementById("cardSpO2");
  const cardRespRate = document.getElementById("cardRespRate");
  const cardTemp = document.getElementById("cardTemp");
  const cardShockIndex = document.getElementById("cardShockIndex");

  const eventStream = document.getElementById("eventStream");
  const canvas = document.getElementById("telemetryCanvas");
  const ctx = canvas.getContext("2d");

  function connect() {
    connectionText.textContent = "Connecting...";
    statusDot.className = "status-dot";

    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      connectionText.textContent = "Live Telemetry Feed";
      statusDot.className = "status-dot connected";
      logEvent("Connected to ICU real-time telemetry stream.", "info");
    };

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        updateDashboard(payload);
      } catch (err) {
        console.error("Payload parse error:", err);
      }
    };

    socket.onclose = () => {
      connectionText.textContent = "Reconnecting...";
      statusDot.className = "status-dot";
      logEvent("Connection lost. Retrying in 2s...", "alert");
      setTimeout(connect, 2000);
    };

    socket.onerror = (err) => {
      console.warn("WebSocket error:", err);
    };
  }

  function updateDashboard(payload) {
    const r = payload.raw_vitals || payload.reading || {};
    const isAnomaly = payload.is_anomaly;
    const sev = payload.severity || "NORMAL";
    const risk = payload.anomaly_score !== undefined ? payload.anomaly_score : 0.0;
    const reasons = payload.flagged_reasons || [];

    // Timestamp
    const ts = r.timestamp || payload.timestamp || (Date.now() / 1000);
    const date = new Date(ts * 1000);
    lastUpdated.textContent = `Last sync: ${date.toLocaleTimeString()}`;

    // Risk Meter & Severity
    riskScoreEl.textContent = risk.toFixed(2);
    riskFill.style.width = `${Math.min(100, Math.round(risk * 100))}%`;

    severityBadge.textContent = sev;

    if (sev === "CRITICAL") {
      severityBadge.className = "severity-badge critical";
      riskFill.style.backgroundColor = "#ef4444";
    } else if (sev === "WARNING") {
      severityBadge.className = "severity-badge";
      severityBadge.style.color = "#f59e0b";
      severityBadge.style.borderColor = "#f59e0b";
      riskFill.style.backgroundColor = "#f59e0b";
    } else {
      severityBadge.className = "severity-badge";
      severityBadge.style.color = "#10b981";
      severityBadge.style.borderColor = "rgba(16, 185, 129, 0.3)";
      riskFill.style.backgroundColor = "#10b981";
    }

    // Vitals Display
    valHeartRate.textContent = r.heart_rate !== undefined ? Math.round(r.heart_rate) : "--";
    valBP.textContent = (r.systolic_bp && r.diastolic_bp) ? `${Math.round(r.systolic_bp)} / ${Math.round(r.diastolic_bp)}` : "-- / --";
    valSpO2.textContent = r.spo2 !== undefined ? `${r.spo2.toFixed(1)}%` : (r.oxygen_saturation ? `${r.oxygen_saturation.toFixed(1)}%` : "--");
    valRespRate.textContent = r.respiratory_rate !== undefined ? Math.round(r.respiratory_rate) : "--";
    valTemp.textContent = r.temperature !== undefined ? r.temperature.toFixed(1) : (r.body_temperature ? r.body_temperature.toFixed(1) : "--");

    // Derived indices
    const sbp = r.systolic_bp || 120;
    const dbp = r.diastolic_bp || 80;
    const pp = sbp - dbp;
    const map = dbp + (pp / 3.0);
    const hr = r.heart_rate || 75;
    const shockIdx = hr / Math.max(sbp, 1.0);

    valMAP.textContent = Math.round(map);
    valShockIndex.textContent = shockIdx.toFixed(2);

    // Alert toggles on cards
    cardHeartRate.classList.toggle("alert", hr > 130 || hr < 45);
    cardBP.classList.toggle("alert", sbp > 170 || sbp < 85);
    const spo2Val = r.spo2 !== undefined ? r.spo2 : r.oxygen_saturation;
    cardSpO2.classList.toggle("alert", spo2Val < 90);
    cardRespRate.classList.toggle("alert", r.respiratory_rate > 26 || r.respiratory_rate < 10);
    const tempVal = r.temperature !== undefined ? r.temperature : r.body_temperature;
    cardTemp.classList.toggle("alert", tempVal > 39.0);
    cardShockIndex.classList.toggle("alert", shockIdx > 0.85);

    // Push into rolling canvas history
    historyPoints.push({
      hr: hr,
      spo2: spo2Val,
      risk: risk,
      is_anomaly: isAnomaly,
    });
    if (historyPoints.length > maxHistory) {
      historyPoints.shift();
    }
    renderChart();

    // Event Log
    if (isAnomaly || sev !== "NORMAL") {
      const reasonStr = reasons.length > 0 ? reasons.join(" | ") : `Risk Score: ${risk.toFixed(2)}`;
      logEvent(`🚨 [${sev}] HR=${Math.round(hr)} SpO2=${spo2Val}% BP=${Math.round(sbp)}/${Math.round(dbp)} - ${reasonStr}`, "alert");
    }
  }

  function renderChart() {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Draw baseline grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 35) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (historyPoints.length < 2) return;

    const step = w / (maxHistory - 1);

    // Draw Heart Rate line
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#38bdf8";

    historyPoints.forEach((pt, i) => {
      const x = i * step;
      // map HR 40-180 to canvas height
      const y = h - ((pt.hr - 40) / 140) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw anomaly alert beacons
    historyPoints.forEach((pt, i) => {
      if (pt.is_anomaly) {
        const x = i * step;
        const y = h - ((pt.hr - 40) / 140) * (h - 20) - 10;

        ctx.fillStyle = "#ef4444";
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = "rgba(239, 68, 68, 0.5)";
        ctx.beginPath();
        ctx.arc(x, y, 11, 0, Math.PI * 2);
        ctx.stroke();
      }
    });
  }

  function logEvent(message, type) {
    const el = document.createElement("div");
    el.className = `event-item ${type === "alert" ? "alert" : ""}`;
    const timestamp = new Date().toLocaleTimeString();
    el.textContent = `[${timestamp}] ${message}`;

    eventStream.insertBefore(el, eventStream.firstChild);
    while (eventStream.children.length > 25) {
      eventStream.removeChild(eventStream.lastChild);
    }
  }

  // Inject Anomaly Button Event
  injectBtn.addEventListener("click", () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ inject_anomaly: true }));
      logEvent("Manual anomaly injection command transmitted.", "alert");
    } else {
      alert("Socket is not open.");
    }
  });

  connect();
})();
