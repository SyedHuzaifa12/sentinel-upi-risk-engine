/* Sentinel design system -- shared vanilla-JS behaviors, loaded on every
   page: entrance choreography, KPI count-up, a Cmd/Ctrl+K command palette,
   toast notifications, and an inline-SVG sparkline drawer. No dependencies. */
(function () {
  "use strict";

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function revealStagger() {
    var els = document.querySelectorAll(".kpi, .data-table tbody tr");
    els.forEach(function (el, i) {
      el.classList.add("reveal");
      el.style.transitionDelay = reduceMotion ? "0ms" : i * 40 + "ms";
    });
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        els.forEach(function (el) { el.classList.add("in"); });
      });
    });
  }

  // Only fires for plain numeric KPI text (queue depth, precision %, etc.) --
  // version strings like "cold-v4" fail the regex and are left untouched.
  function countUp() {
    if (reduceMotion) return;
    document.querySelectorAll(".kpi-value").forEach(function (el) {
      var text = el.textContent.trim();
      var m = text.match(/^([^\d-]*)(-?\d+(?:\.\d+)?)(.*)$/);
      if (!m) return;
      var prefix = m[1], target = parseFloat(m[2]), suffix = m[3];
      var decimals = (m[2].split(".")[1] || "").length;
      var start = null, duration = 600;
      function frame(ts) {
        if (start === null) start = ts;
        var t = Math.min(1, (ts - start) / duration);
        var eased = 1 - Math.pow(1 - t, 3);
        el.textContent = prefix + (target * eased).toFixed(decimals) + suffix;
        if (t < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    });
  }

  function ensureToastStack() {
    var stack = document.querySelector(".toast-stack");
    if (!stack) {
      stack = document.createElement("div");
      stack.className = "toast-stack";
      document.body.appendChild(stack);
    }
    return stack;
  }

  function showToast(message, type) {
    var stack = ensureToastStack();
    var toast = document.createElement("div");
    toast.className = "toast" + (type ? " " + type : "");
    var icon = type === "error" ? "fa-circle-exclamation" : type === "success" ? "fa-circle-check" : "fa-circle-info";
    toast.innerHTML = '<i class="fas ' + icon + '"></i><span></span>';
    toast.querySelector("span").textContent = message;
    stack.appendChild(toast);
    requestAnimationFrame(function () { toast.classList.add("show"); });
    setTimeout(function () {
      toast.classList.remove("show");
      setTimeout(function () { toast.remove(); }, 300);
    }, 4000);
  }

  // Real inline SVG line + gradient-fill area, no chart library.
  function drawSparkline(containerId, values) {
    var el = document.getElementById(containerId);
    if (!el || !values.length) return;
    var w = 600, h = 64, pad = 4;
    var max = Math.max.apply(null, values), min = Math.min.apply(null, values);
    var range = max - min || 1;
    var step = (w - pad * 2) / (values.length - 1 || 1);
    var points = values.map(function (v, i) {
      return [pad + i * step, pad + (h - pad * 2) * (1 - (v - min) / range)];
    });
    var line = points.map(function (p, i) { return (i === 0 ? "M" : "L") + p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" ");
    var area = line + " L" + points[points.length - 1][0].toFixed(1) + "," + (h - pad) +
      " L" + points[0][0].toFixed(1) + "," + (h - pad) + " Z";
    var gradId = "sparkfill-" + containerId;
    el.innerHTML =
      '<svg viewBox="0 0 ' + w + " " + h + '" preserveAspectRatio="none">' +
      '<defs><linearGradient id="' + gradId + '" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" style="stop-color:var(--accent);stop-opacity:0.35"/>' +
      '<stop offset="100%" style="stop-color:var(--accent);stop-opacity:0"/>' +
      "</linearGradient></defs>" +
      '<path d="' + area + '" style="fill:url(#' + gradId + ');stroke:none"/>' +
      '<path d="' + line + '" style="fill:none;stroke:var(--accent);stroke-width:2;stroke-linejoin:round;stroke-linecap:round"/>' +
      "</svg>";
  }

  var ROUTES = [
    { label: "Monitoring", icon: "fa-chart-line", href: "/monitoring/" },
    { label: "Review queue", icon: "fa-magnifying-glass", href: "/review/" },
    { label: "Sandbox", icon: "fa-flask", href: "/prediction/" },
    { label: "API docs", icon: "fa-code", href: "/api/docs" },
    { label: "View source", icon: "fa-github", href: "https://github.com/SyedHuzaifa12/sentinel-upi-risk-engine" },
  ];

  function initCommandPalette() {
    var overlay = null, panel = null, input = null, activeIndex = 0;

    function buildItems(query) {
      var items = ROUTES.slice();
      var replayBtn = document.getElementById("replay-btn");
      if (replayBtn && !replayBtn.disabled) {
        items.push({ label: replayBtn.textContent.trim(), icon: "fa-play", action: function () { replayBtn.click(); } });
      }
      if (!query) return items;
      var q = query.toLowerCase();
      return items.filter(function (it) { return it.label.toLowerCase().indexOf(q) !== -1; });
    }

    function activate(it) {
      close();
      if (it.action) it.action();
      else window.location.href = it.href;
    }

    function renderList() {
      var list = buildItems(input.value);
      var listEl = panel.querySelector(".cmdk-list");
      listEl.innerHTML = "";
      if (!list.length) {
        listEl.innerHTML = '<div class="cmdk-empty">No matches</div>';
        return;
      }
      list.forEach(function (it, i) {
        var div = document.createElement("div");
        div.className = "cmdk-item" + (i === activeIndex ? " active" : "");
        div.innerHTML = '<i class="fas ' + it.icon + '"></i><span></span>';
        div.querySelector("span").textContent = it.label;
        div.addEventListener("click", function () { activate(it); });
        listEl.appendChild(div);
      });
      return list;
    }

    function close() {
      if (!overlay) return;
      overlay.remove();
      overlay = null;
    }

    function open() {
      if (overlay) return;
      activeIndex = 0;
      overlay = document.createElement("div");
      overlay.className = "cmdk-overlay";
      overlay.innerHTML =
        '<div class="cmdk-panel"><input class="cmdk-input" placeholder="Jump to..." autocomplete="off">' +
        '<div class="cmdk-list"></div><div class="cmdk-hint">' +
        "<span><kbd>&uarr;&darr;</kbd> navigate</span><span><kbd>Enter</kbd> select</span><span><kbd>Esc</kbd> close</span>" +
        "</div></div>";
      overlay.addEventListener("click", function (e) { if (e.target === overlay) close(); });
      document.body.appendChild(overlay);
      panel = overlay.querySelector(".cmdk-panel");
      input = overlay.querySelector(".cmdk-input");
      input.addEventListener("input", function () { activeIndex = 0; renderList(); });
      renderList();
      input.focus();
    }

    document.addEventListener("keydown", function (e) {
      var isMod = e.metaKey || e.ctrlKey;
      if (isMod && e.key.toLowerCase() === "k") { e.preventDefault(); overlay ? close() : open(); return; }
      if (!overlay) return;
      var list = buildItems(input.value);
      if (e.key === "Escape") close();
      else if (e.key === "ArrowDown") { e.preventDefault(); activeIndex = Math.min(list.length - 1, activeIndex + 1); renderList(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); activeIndex = Math.max(0, activeIndex - 1); renderList(); }
      else if (e.key === "Enter") { e.preventDefault(); if (list[activeIndex]) activate(list[activeIndex]); }
    });
  }

  window.SentinelToast = { show: showToast };
  window.SentinelSparkline = drawSparkline;

  document.addEventListener("DOMContentLoaded", function () {
    revealStagger();
    countUp();
    initCommandPalette();
  });
})();
