/* Renders figure panels from window.FIGS into <div class="fig" data-fig="name"> elements.
   Each figure is a list of matplotlib axes exported as data (lines, bars, reference lines, text). */
(function () {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  let W = 560, H = 320;
  const DASH = { "--": "5 4", ":": "1.5 3", "-.": "6 3 1.5 3", "-": null };

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function scale(kind, lim, a, b, sp) {
    let f;
    if (kind === "log") f = v => Math.log10(Math.max(v, 1e-12));
    else if (kind === "symlog") {
      const lt = (sp && sp.linthresh) || 1, base = (sp && sp.base) || 10, ls = (sp && sp.linscale) || 1;
      const adj = ls / (1 - 1 / base);
      f = v => {
        const s = Math.sign(v), x = Math.abs(v);
        return x <= lt ? s * (x / lt) * adj : s * (adj + Math.log(x / lt) / Math.log(base));
      };
    } else f = v => v;
    const f0 = f(lim[0]), f1 = f(lim[1]);
    const g = v => a + (f(v) - f0) / (f1 - f0) * (b - a);
    g.inv = null;
    return g;
  }

  function fmt(v) {
    if (v === null || v === undefined || !isFinite(v)) return "n/a";
    const a = Math.abs(v);
    if (a >= 1e5) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (a >= 100) return v.toLocaleString("en-US", { maximumFractionDigits: 1 });
    if (a >= 1) return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
    return v.toPrecision(3);
  }

  function textWidth(s, px) { return (s || "").length * px * 0.62; }

  function panel(container, ax, twin) {
    const card = document.createElement("div");
    card.className = "panel";
    if (ax.titre) {
      const t = document.createElement("div");
      t.className = "ptitle";
      t.textContent = ax.titre;
      card.appendChild(t);
    }
    // margins
    const yLabels = (ax.yt || []).map(t => t[1]);
    const maxY = Math.max(0, ...yLabels.map(s => textWidth(s, 11)));
    let left = Math.min(240, Math.max(44, maxY + 18)) + (ax.yl ? 16 : 0);
    const right = twin ? 52 : 14;
    // long category labels on x: rotate them and reserve room below and on the left
    const xin0 = v => v >= Math.min(...ax.xlim) - 1e-9 && v <= Math.max(...ax.xlim) + 1e-9;
    const xt0 = (ax.xt || []).filter(t => xin0(t[0]));
    const maxXW = Math.max(0, ...xt0.map(t => textWidth(t[1], 10.5)));
    const slot = (W - left - right) / Math.max(1, xt0.length);
    const longX = maxXW > slot - 6;
    const COS = 0.82, SIN = 0.57;
    if (longX) left = Math.max(left, Math.min(220, maxXW * COS - slot / 2 + 10));
    const extra = longX ? Math.round(maxXW * SIN) : 0;
    const Hl = H + extra;
    const top = 12, bottom = (ax.xl ? 46 : 30) + extra;
    const x0 = left, x1 = W - right, y0 = Hl - bottom, y1 = top;
    const sx = scale(ax.xs, ax.xlim, x0, x1, ax.xsp);
    const sy = scale(ax.ys, ax.ylim, y0, y1, null);

    const svg = el("svg", { viewBox: `0 0 ${W} ${Hl}`, class: "chart", role: "img", "aria-label": ax.titre || "chart" });
    const clipId = "c" + Math.random().toString(36).slice(2, 9);
    const defs = el("defs", {}, svg);
    el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1 }, el("clipPath", { id: clipId }, defs));

    // grid + ticks
    const gGrid = el("g", { class: "grid" }, svg);
    const xin = v => v >= Math.min(...ax.xlim) - 1e-9 && v <= Math.max(...ax.xlim) + 1e-9;
    const yin = (v, lim) => v >= Math.min(...lim) - 1e-9 && v <= Math.max(...lim) + 1e-9;
    const xt = xt0;
    xt.forEach(t => {
      const X = sx(t[0]);
      el("line", { x1: X, x2: X, y1: y1, y2: y0, class: "gl" }, gGrid);
      const tx = el("text", { x: longX ? X + 3 : X, y: y0 + 15, class: "tick", "text-anchor": longX ? "end" : "middle" }, svg);
      if (longX) tx.setAttribute("transform", `rotate(-35 ${X + 3} ${y0 + 12})`);
      tx.textContent = t[1];
    });
    (ax.yt || []).filter(t => yin(t[0], ax.ylim)).forEach(t => {
      const Y = sy(t[0]);
      el("line", { x1: x0, x2: x1, y1: Y, y2: Y, class: "gl" }, gGrid);
      const ty = el("text", { x: x0 - 6, y: Y + 3.5, class: "tick", "text-anchor": "end" }, svg);
      ty.textContent = t[1];
    });
    el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1, class: "frame" }, svg);
    if (ax.xl) {
      const l = el("text", { x: (x0 + x1) / 2, y: Hl - 6, class: "alabel", "text-anchor": "middle" }, svg);
      l.textContent = ax.xl;
    }
    if (ax.yl) {
      const l = el("text", { x: 12, y: (y0 + y1) / 2, class: "alabel", "text-anchor": "middle", transform: `rotate(-90 12 ${(y0 + y1) / 2})` }, svg);
      l.textContent = ax.yl;
    }

    const legend = [];
    const lines = [];
    function draw(axis, sxx, syy, gp) {
      axis.s.forEach(s => {
        const col = Array.isArray(s.c) ? s.c[0] : s.c;
        const op = s.a === undefined ? 1 : s.a;
        if (s.t === "l") {
          let d = "", pen = false;
          const pts = [];
          for (let i = 0; i < s.x.length; i++) {
            const xv = s.x[i], yv = s.y[i];
            if (xv === null || yv === null || (ax.xs === "log" && xv <= 0)) { pen = false; continue; }
            const X = sxx(xv), Y = syy(yv);
            d += (pen ? "L" : "M") + X.toFixed(1) + " " + Y.toFixed(1);
            pen = true;
            pts.push([X, Y, xv, yv]);
          }
          el("path", { d, fill: "none", stroke: col, "stroke-width": Math.max(1.2, (s.lw || 1.5) * 0.95), "stroke-dasharray": DASH[s.ls] || null, "stroke-linejoin": "round", opacity: op }, gp);
          if (s.m && pts.length <= 40) pts.forEach(p => el("circle", { cx: p[0], cy: p[1], r: 2.6, fill: col, opacity: op }, gp));
          if (s.n && !s.n.startsWith("_")) legend.push({ n: s.n, c: col, ls: s.ls, k: "l" });
          lines.push({ n: s.n || "", c: col, pts });
        } else if (s.t === "bar" || s.t === "barh") {
          const cols = Array.isArray(s.c) ? s.c : null;
          for (let i = 0; i < s.x.length; i++) {
            let xa = s.x[i], xb = s.x[i] + s.w[i], ya = s.y[i], yb = s.y[i] + s.h[i];
            if (ax.xs === "log") { xa = Math.max(xa, Math.min(...ax.xlim)); xb = Math.max(xb, Math.min(...ax.xlim)); }
            const X0 = sxx(Math.min(xa, xb)), X1 = sxx(Math.max(xa, xb)), Y0 = syy(Math.max(ya, yb)), Y1 = syy(Math.min(ya, yb));
            const r = el("rect", { x: Math.min(X0, X1), y: Math.min(Y0, Y1), width: Math.max(0.5, Math.abs(X1 - X0)), height: Math.max(0.5, Math.abs(Y1 - Y0)), fill: cols ? cols[i] : col, opacity: op * 0.92, class: "bar" }, gp);
            const val = s.t === "bar" ? s.h[i] : s.w[i];
            const tt = el("title", {}, r);
            tt.textContent = (s.n ? s.n + ": " : "") + fmt(val);
          }
          if (s.n && !s.n.startsWith("_")) legend.push({ n: s.n, c: col, k: "b" });
        } else if (s.t === "h") {
          el("line", { x1: x0, x2: x1, y1: syy(s.y), y2: syy(s.y), stroke: col, "stroke-width": 1, "stroke-dasharray": DASH[s.ls] || null, opacity: 0.9 }, gp);
          if (s.n && !s.n.startsWith("_")) legend.push({ n: s.n, c: col, ls: s.ls, k: "l" });
        } else if (s.t === "v") {
          el("line", { x1: sxx(s.x), x2: sxx(s.x), y1: y1, y2: y0, stroke: col, "stroke-width": 1, "stroke-dasharray": DASH[s.ls] || null, opacity: 0.9 }, gp);
          if (s.n && !s.n.startsWith("_")) legend.push({ n: s.n, c: col, ls: s.ls, k: "l" });
        } else if (s.t === "poly") {
          const d = s.x.map((xv, i) => (i ? "L" : "M") + sxx(xv).toFixed(1) + " " + syy(s.y[i]).toFixed(1)).join("") + "Z";
          el("path", { d, fill: col, opacity: Math.min(0.55, op), stroke: "none" }, gp);
        } else if (s.t === "txt") {
          const X = s.axe ? x0 + s.x * (x1 - x0) : sxx(s.x), Y = s.axe ? y0 - s.y * (y0 - y1) : syy(s.y);
          const t = el("text", { x: s.ha === "center" ? X : X + 2, y: Y + (s.va === "center" ? 3.5 : s.va === "bottom" ? -2 : 0), class: "ann", style: s.c && s.c !== "#a3a8ae" ? "fill:" + s.c : null, "text-anchor": s.ha === "right" ? "end" : s.ha === "center" ? "middle" : "start" }, svg);
          if (s.ha !== "center" && s.ha !== "right" && X + 2 + textWidth(s.s, 10) > W - 2) { t.setAttribute("x", W - 2); t.setAttribute("text-anchor", "end"); }
          t.textContent = s.s;
        }
      });
    }
    const gMain = el("g", { "clip-path": `url(#${clipId})` }, svg);
    draw(ax, sx, sy, gMain);

    if (twin) {
      const sy2 = scale(twin.ys, twin.ylim, y0, y1, null);
      (twin.yt || []).filter(t => yin(t[0], twin.ylim)).forEach(t => {
        const ty = el("text", { x: x1 + 6, y: sy2(t[0]) + 3.5, class: "tick", "text-anchor": "start" }, svg);
        ty.textContent = t[1];
      });
      draw(twin, sx, sy2, el("g", { "clip-path": `url(#${clipId})` }, svg));
    }

    // hover readout for line charts
    if (lines.length && lines.some(l => l.pts.length)) {
      const cross = el("line", { y1: y1, y2: y0, class: "cross", visibility: "hidden" }, svg);
      const tip = document.createElement("div");
      tip.className = "tip";
      card.style.position = "relative";
      const hit = el("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", ev => {
        const box = svg.getBoundingClientRect();
        const X = (ev.clientX - box.left) * svg.viewBox.baseVal.width / box.width;
        let best = null;
        const rows = [];
        lines.forEach(l => {
          if (!l.pts.length) return;
          let lo = 0, hi = l.pts.length - 1;
          let k = 0, dk = Infinity;
          for (let i = lo; i <= hi; i++) { const dd = Math.abs(l.pts[i][0] - X); if (dd < dk) { dk = dd; k = i; } }
          if (dk < 30) {
            rows.push([l.n, l.c, l.pts[k][3]]);
            if (!best || dk < best[1]) best = [l.pts[k], dk];
          }
        });
        if (!best) { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); return; }
        cross.setAttribute("x1", best[0][0]); cross.setAttribute("x2", best[0][0]);
        cross.setAttribute("visibility", "visible");
        tip.innerHTML = `<div class="tx">${fmt(best[0][2])}</div>` + rows.slice(0, 8).map(r => `<div><i style="background:${r[1]}"></i>${r[0] ? r[0] + ": " : ""}<b>${fmt(r[2])}</b></div>`).join("");
        tip.style.display = "block";
        const cb = card.getBoundingClientRect();
        let L = ev.clientX - cb.left + 14;
        if (L + 220 > cb.width) L = ev.clientX - cb.left - 14 - Math.min(220, tip.offsetWidth || 180);
        tip.style.left = Math.max(4, L) + "px";
        tip.style.top = (ev.clientY - cb.top + 10) + "px";
      });
      hit.addEventListener("mouseleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
      card.appendChild(tip);
    }

    card.appendChild(svg);
    if (legend.length) {
      const lg = document.createElement("div");
      lg.className = "legend";
      const seen = new Set();
      legend.forEach(it => {
        if (seen.has(it.n)) return;
        seen.add(it.n);
        const s = document.createElement("span");
        const dash = it.k === "l" && it.ls && it.ls !== "-" ? " dashed" : "";
        s.innerHTML = `<i class="${it.k === "b" ? "sw" : "ln" + dash}" style="--c:${it.c}"></i>`;
        s.appendChild(document.createTextNode(it.n));
        lg.appendChild(s);
      });
      card.appendChild(lg);
    }
    container.appendChild(card);
  }

  function render(div) {
    const f = window.FIGS && window.FIGS[div.dataset.fig];
    if (!f) return;
    const grid = document.createElement("div");
    grid.className = "panels";
    const axes = f.axes;
    const only = div.dataset.panels ? div.dataset.panels.split(",").map(Number) : null;
    axes.forEach((ax, i) => {
      if (ax.droite) return;
      if (only && !only.includes(i)) return;
      const twin = axes.find((b, j) => j !== i && b.droite && b.pos.every((v, k) => Math.abs(v - ax.pos[k]) < 1e-6));
      W = f.vw || 560; H = f.vh || 320;
      panel(grid, ax, twin || null);
    });
    if (grid.children.length === 1) grid.classList.add("single");
    div.prepend(grid);
  }

  function init() { document.querySelectorAll(".fig[data-fig]").forEach(render); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
