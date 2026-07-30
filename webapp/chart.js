/* GSF Explorer — reactive SVG spectrum chart.
 *
 * Pure view over raw model grids: x^gamma weighting, log/linear, visibility,
 * bands, hover crosshair, box-zoom and pan all happen here in JS, so display
 * controls respond instantly with no Python round-trip.
 *
 * Renders one or more models (parameter sets) on shared axes: model 0 is
 * solid with bands; further models are dotted / dash-dot, groups + total
 * only, no bands.
 */

import { html, useState, useRef, useMemo }
  from "https://cdn.jsdelivr.net/npm/htm@3.1.1/preact/standalone.module.js";

/* Series palettes validated per surface (dataviz six checks, adjacent PASS) */
export const THEMES = {
  dark: {
    surface: "#0d1117",
    ink: "#e9edf4", ink2: "#98a1b3", ink3: "#5b6472",
    hair: "rgba(233,237,244,0.10)", hair2: "rgba(233,237,244,0.22)",
    accent: "#a08dfa", accentDim: "rgba(160,141,250,0.14)",
    groups: { "p": "#e64d6e", "He": "#c08a00", "O*": "#1a9e70", "Fe*": "#3d8ce0" },
    elements: ["#a5b0c2", "#b18cff", "#7d8590", "#d0a9ff", "#8e9dbb", "#c7b9ff"],
    glow: 0.22,
  },
  light: {
    surface: "#f6f7f9",
    ink: "#1a2130", ink2: "#4d5668", ink3: "#8892a3",
    hair: "rgba(26,33,48,0.09)", hair2: "rgba(26,33,48,0.24)",
    accent: "#6c58e8", accentDim: "rgba(108,88,232,0.12)",
    groups: { "p": "#c73558", "He": "#ab8200", "O*": "#006e42", "Fe*": "#2a6fc0" },
    elements: ["#5c6880", "#7a5fd0", "#98a1b3", "#9d7fe0", "#66738c", "#8f7fd8"],
    glow: 0.10,
  },
};

export const GROUPS = ["p", "He", "O*", "Fe*"];
/* full evaluation domain in decades: 0.1 – 10^11 GeV/GV. Below-validity
   regions (LIS < 1 GV, sub-threshold total energy) come back as zeros and
   render as gaps. */
export const X_DOMAIN = [-1, 11];
/* line style per model slot (identity of a model = its dash, not a color) */
export const MODEL_DASH = ["none", "2 3.5", "9 3 2.5 3"];

const log10 = Math.log10;

/* ---------------------------------------------------------------- format */

export function sciLabel(v) {
  if (!Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const e = Math.floor(log10(Math.abs(v)));
  if (e >= -2 && e <= 3) return v.toPrecision(3).replace(/\.?0+$/, "");
  return `${(v / 10 ** e).toFixed(2)}×10${sup(e)}`;
}

const SUP = { "-": "⁻", ".": "·", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴",
              5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
export const sup = (n) => String(n).split("").map((c) => SUP[c] ?? c).join("");

/* ----------------------------------------------------------------- paths */

function segments(ys) {
  const runs = [];
  let run = null;
  for (let i = 0; i < ys.length; i++) {
    if (Number.isFinite(ys[i])) {
      (run ??= []).push(i);
    } else if (run) { runs.push(run); run = null; }
  }
  if (run) runs.push(run);
  return runs;
}

function linePath(px, py) {
  return segments(py)
    .map((run) => "M" + run.map((i) => `${px[i].toFixed(1)},${py[i].toFixed(1)}`).join("L"))
    .join("");
}

/* Band between lo and hi pixel arrays. Where the lower edge is undefined
 * (flux − σ ≤ 0 on a log axis: "consistent with zero"), it clamps to the
 * plot bottom instead of collapsing onto the upper edge — this was the
 * high-energy artifact. */
function bandPath(px, lo, hi, bottomY) {
  return segments(hi)
    .map((run) => {
      const up = run.map((i) => `${px[i].toFixed(1)},${hi[i].toFixed(1)}`).join("L");
      const dn = [...run].reverse()
        .map((i) => `${px[i].toFixed(1)},${(Number.isFinite(lo[i]) ? lo[i] : bottomY).toFixed(1)}`)
        .join("L");
      return `M${up}L${dn}Z`;
    })
    .join("");
}

/* ----------------------------------------------------------------- ticks */

/* Log-axis ticks, responsive to zoom: majors at decades (stepped by label
 * budget); minor 2–9 ticks appear once a decade is wide enough on screen;
 * when the window is narrower than ~1.3 decades the minors get their own
 * "k×10ⁿ" labels (a deep zoom may contain no decade line at all). */
function logTicks(lo, hi, pxLen = 800, labelPx = 38) {
  const span = Math.max(hi - lo, 1e-9);
  const pxPerDecade = pxLen / span;
  const step = Math.max(1, Math.ceil((span * labelPx) / Math.max(pxLen, 60)));
  const ticks = [];
  for (let d = Math.ceil(lo); d <= Math.floor(hi) + 1e-9; d += step)
    ticks.push(d);

  const minors = [];
  const minorLabels = [];
  if (step === 1 && pxPerDecade >= 34) {
    let subs = null;   // which minors get labels
    if (span <= 2.5)
      subs = pxPerDecade >= 2200 ? [2, 3, 4, 5, 6, 7, 8, 9]
           : pxPerDecade >= 900 ? [2, 3, 5, 7] : [2, 5];
    for (let d = Math.floor(lo); d < hi; d++)
      for (let k = 2; k <= 9; k++) {
        const v = d + log10(k);
        if (v <= lo || v >= hi) continue;
        minors.push(v);
        if (subs?.includes(k))
          minorLabels.push({ v, text: `${k}×10${sup(d)}` });
      }
  }
  return { ticks, minors, minorLabels, step };
}

/* Linear ticks with density-gated fifth-step minors. */
function linTicks(lo, hi, pxLen = 800) {
  const raw = (hi - lo) / 5;
  const mag = 10 ** Math.floor(log10(Math.max(raw, 1e-300)));
  const step = [1, 2, 5, 10].map((s) => s * mag).find((s) => (hi - lo) / s <= 6);
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step)
    ticks.push(v);
  const minors = [];
  if ((pxLen / ((hi - lo) / step)) / 5 >= 8) {
    const ms = step / 5;
    for (let v = Math.ceil(lo / ms) * ms; v <= hi + ms * 1e-6; v += ms)
      if (Math.abs(v / step - Math.round(v / step)) > 1e-9 && v > lo && v < hi)
        minors.push(v);
  }
  return { ticks, minors, minorLabels: [], step };
}

/* ------------------------------------------------------------- component */

export function Chart({ models, view, theme, mode, xWindow, hoverEnabled = true,
                        width, height, insets, onWindow, onHome, onHoverInfo }) {
  const [hover, setHover] = useState(null);
  const [drag, setDrag] = useState(null);   // {x0,y0,x1,y1}
  const svgRef = useRef(null);

  const m = {
    l: insets.l + 74,
    r: insets.r + 26,
    t: insets.t + 18, b: insets.b + 58,
  };
  const pw = Math.max(width - m.l - m.r, 40);
  const ph = Math.max(height - m.t - m.b, 40);

  const built = useMemo(() => {
    if (!models?.length || !models[0].data) return null;
    const x = models[0].data.x;
    const g = view.gamma;
    const rows = [];
    for (let mi = 0; mi < models.length; mi++) {
      const { data, version } = models[mi];
      const tag = models.length > 1 ? version : null;
      const withBand = mi === 0 || view.overlayBands;
      /* flux view: y = Φ·xᵞ; ratio view: y = Φᵢ/Φ_total (γ cancels; band
         edges use σᵢ/Φ_total — correlation with the total neglected) */
      const tot = data.total?.flux;
      const weigh = view.ratio
        ? (f) => f.map((v, i) =>
            (v > 0 && tot?.[i] > 0 ? v / tot[i] : NaN))
        : (f) => f.map((v, i) => (v > 0 ? v * x[i] ** g : NaN));
      if (view.showTotal && data.total && !view.ratio) {
        const { flux, err } = data.total;
        rows.push({
          name: "all-particle", tag, mi, kind: "total", color: theme.ink,
          w: weigh(flux),
          lo: withBand && err ? weigh(flux.map((v, i) => v - err[i])) : null,
          hi: withBand && err ? weigh(flux.map((v, i) => v + err[i])) : null,
        });
      }
      let ei = 0;
      for (const s of data.series) {
        const isGroup = s.name in theme.groups;
        if (!isGroup && mi > 0) continue;   // elements: primary model only
        if (!view.visible[s.name]) { if (!isGroup) ei++; continue; }
        const color = isGroup ? theme.groups[s.name]
          : theme.elements[ei++ % theme.elements.length];
        rows.push({
          name: s.name, tag, mi, kind: isGroup ? "group" : "element", color,
          w: weigh(s.flux),
          lo: withBand && isGroup && s.err
            ? weigh(s.flux.map((v, i) => Math.max(v - s.err[i], 0))) : null,
          hi: withBand && isGroup && s.err
            ? weigh(s.flux.map((v, i) => v + s.err[i])) : null,
        });
      }
    }
    return { x, rows };
  }, [models, view.gamma, view.ratio, view.showTotal, view.visible,
      view.overlayBands, theme]);

  const scale = useMemo(() => {
    if (!built) return null;
    /* the view window follows the controls instantly; the data grid catches
       up asynchronously (re-grid refines resolution, display never waits) */
    const lgx0 = xWindow ? xWindow[0] : log10(built.x[0]);
    const lgx1 = xWindow ? xWindow[1] : log10(built.x[built.x.length - 1]);
    const X = (v) => m.l + ((log10(v) - lgx0) / (lgx1 - lgx0)) * pw;

    let ylo, yhi;
    if (view.yRange) {
      [ylo, yhi] = view.ylog
        ? [log10(view.yRange[0]), log10(view.yRange[1])]
        : view.yRange;
    } else {
      const solid = built.rows.filter((r) => r.kind !== "element");
      const rows = solid.length ? solid : built.rows;
      /* autoscale from the mean curves WITHIN the visible x-window, so a
         zoomed view rescales to what is on screen (full-grid fallback in
         case the window holds no positive point) */
      const lgs = built.x.map(log10);
      const take = (test) => rows.flatMap((r) =>
        r.w.filter((v, i) => Number.isFinite(v) && v > 0 && test(lgs[i])));
      let pool = take((l) => l >= lgx0 - 1e-9 && l <= lgx1 + 1e-9);
      if (!pool.length) pool = take(() => true);
      if (!pool.length) return null;
      if (view.ylog) {
        ylo = log10(Math.min(...pool) / 2.2);
        yhi = log10(Math.max(...pool) * 2.2);
      } else { ylo = 0; yhi = Math.max(...pool) * 1.06; }
    }
    const Y = view.ylog
      ? (v) => (v > 0 ? m.t + ph - ((log10(v) - ylo) / (yhi - ylo)) * ph : NaN)
      : (v) => m.t + ph - ((v - ylo) / (yhi - ylo)) * ph;
    const Yinv = view.ylog
      ? (py) => 10 ** (ylo + ((m.t + ph - py) / ph) * (yhi - ylo))
      : (py) => ylo + ((m.t + ph - py) / ph) * (yhi - ylo);
    return { X, Y, Yinv, lgx0, lgx1, ylo, yhi };
  }, [built, view.ylog, view.yRange, xWindow, pw, ph, m.l, m.t]);

  if (!built || !scale)
    return html`<svg width=${width} height=${height} class="chart"></svg>`;

  const { X, Y, Yinv, lgx0, lgx1, ylo, yhi } = scale;
  const px = built.x.map(X);
  const bottomY = m.t + ph;
  const shapes = built.rows.map((r) => ({
    ...r,
    py: r.w.map(Y),
    band: view.showBands && r.hi
      ? bandPath(px, r.lo.map(Y), r.hi.map(Y), bottomY) : null,
  }));

  /* legend (top-right box): primary components, then model line styles */
  const legendRows = shapes.filter((s) => s.mi === 0).map((s) => ({
    text: s.name, color: s.color,
    dash: s.kind === "element" ? "5 3.5" : "none",
    lw: s.kind === "total" ? 2.6 : 2,
  }));
  if (models.length > 1) {
    legendRows.push(null);   // separator
    models.forEach((mm, mi) => legendRows.push({
      text: mm.version, color: theme.ink, dash: MODEL_DASH[mi], lw: 1.8,
    }));
  }
  const LG = { rowH: 17, pad: 10, sample: 20, gap: 7 };
  const lgW = LG.pad * 2 + LG.sample + LG.gap + Math.max(...legendRows.map(
    (r) => (r ? r.text.length : 0))) * 6.6;
  const lgH = LG.pad * 2 - 4 +
    legendRows.reduce((a, r) => a + (r ? LG.rowH : 8), 0);
  const lgX = m.l + pw - lgW - 10, lgY = m.t + 10;

  /* axes */
  const xt = logTicks(lgx0, lgx1, pw, 38);
  const yAxis = view.ylog ? logTicks(ylo, yhi, ph, 22)
                          : linTicks(ylo, yhi, ph);
  const yTickY = (t) => (view.ylog ? Y(10 ** t) : Y(t));

  /* pointer handling */
  const rel = (e) => {
    const r = svgRef.current.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const toIdx = (xpx) => {
    const lg = lgx0 + ((xpx - m.l) / pw) * (lgx1 - lgx0);
    const n = built.x.length;
    const i = Math.round(((lg - log10(built.x[0])) /
      (log10(built.x[n - 1]) - log10(built.x[0]))) * (n - 1));
    return Math.max(0, Math.min(n - 1, i));
  };
  const commitWindow = (d) => {
    const lg = (xpx) => lgx0 + ((xpx - m.l) / pw) * (lgx1 - lgx0);
    /* a drag that started on an axis moves that coordinate ONLY; the other
       one is frozen at its current displayed range */
    const wantX = d.axis !== "y", wantY = d.axis !== "x";
    const yFrozen = view.ylog ? [10 ** ylo, 10 ** yhi] : [ylo, yhi];
    if (mode === "pan") {
      let ddec = wantX ? lg(d.x0) - lg(d.x1) : 0;
      ddec = Math.max(X_DOMAIN[0] - lgx0, Math.min(X_DOMAIN[1] - lgx1, ddec));
      const dy = wantY ? (d.y1 - d.y0) / ph * (yhi - ylo) : 0;
      const nylo = ylo + dy, nyhi = yhi + dy;
      onWindow?.({
        dmin: +(lgx0 + ddec).toFixed(2), dmax: +(lgx1 + ddec).toFixed(2),
        yRange: view.ylog ? [10 ** nylo, 10 ** nyhi] : [nylo, nyhi],
      });
    } else {
      if ((wantX && Math.abs(d.x1 - d.x0) < 18) ||
          (wantY && Math.abs(d.y1 - d.y0) < 14)) return;
      const a = lg(Math.max(Math.min(d.x0, d.x1), m.l));
      const b = lg(Math.min(Math.max(d.x0, d.x1), m.l + pw));
      const vTop = Yinv(Math.max(Math.min(d.y0, d.y1), m.t));
      const vBot = Yinv(Math.min(Math.max(d.y0, d.y1), m.t + ph));
      onWindow?.({
        dmin: wantX ? Math.max(X_DOMAIN[0], Math.floor(a * 20) / 20)
                    : +lgx0.toFixed(3),
        dmax: wantX ? Math.min(X_DOMAIN[1], Math.ceil(b * 20) / 20)
                    : +lgx1.toFixed(3),
        yRange: wantY ? [vBot, vTop] : yFrozen,
      });
    }
  };

  const onMove = (e) => {
    if (drag) {
      const p = rel(e);
      setDrag({ ...drag, x1: p.x, y1: p.y });
      return;
    }
    if (!hoverEnabled) return;
    const p = rel(e);
    const i = toIdx(p.x);
    setHover(i);
    onHoverInfo?.({
      i, x: built.x[i],
      rows: shapes.map((s) => ({ name: s.name, tag: s.tag, mi: s.mi,
                                 color: s.color, v: s.w[i] }))
        .filter((r) => Number.isFinite(r.v))
        .sort((a, b) => b.v - a.v),
      pxX: px[i],
    });
  };
  const clearHover = () => { setHover(null); onHoverInfo?.(null); };
  const onDown = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = rel(e);
    setHover(null); onHoverInfo?.(null);
    /* start zone decides the gesture: x-axis strip -> x only, y-axis strip
       -> y only, canvas -> both (trackpad gestures always move both) */
    const axis =
      p.y > m.t + ph && p.x >= m.l && p.x <= m.l + pw ? "x"
      : p.x < m.l && p.y >= m.t && p.y <= m.t + ph ? "y" : null;
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y, axis });
  };
  const onUp = () => {
    if (drag && (drag.x0 !== drag.x1 || drag.y0 !== drag.y1))
      commitWindow(drag);
    setDrag(null);
  };

  /* trackpad: pinch (wheel + ctrlKey on macOS) zooms about the cursor,
     two-finger scroll pans. Events are normalized for deltaMode (line/page
     wheels arrive in non-pixel units) and coalesced per animation frame so
     momentum scrolling stays at native speed instead of queueing renders. */
  const wheelAcc = useRef(null);
  const flushWheel = () => {
    const a = wheelAcc.current;
    wheelAcc.current = null;
    if (!a) return;
    let x0 = lgx0, x1 = lgx1, lo = ylo, hi = yhi;
    if (a.pinch) {
      const f = Math.exp(Math.max(-80, Math.min(80, a.pinch)) * 0.0035);
      const lgc = x0 + ((a.x - m.l) / pw) * (x1 - x0);
      x0 = lgc - (lgc - x0) * f;
      x1 = lgc + (x1 - lgc) * f;
      const yc = lo + ((m.t + ph - a.y) / ph) * (hi - lo);
      lo = yc - (yc - lo) * f;
      hi = yc + (hi - yc) * f;
    }
    if (a.dx || a.dy) {
      let ddec = (a.dx / pw) * (x1 - x0);
      ddec = Math.max(X_DOMAIN[0] - x0, Math.min(X_DOMAIN[1] - x1, ddec));
      x0 += ddec; x1 += ddec;
      const dy = (a.dy / ph) * (hi - lo);
      lo += dy; hi += dy;
    }
    x0 = Math.max(X_DOMAIN[0], x0);
    x1 = Math.min(X_DOMAIN[1], x1);
    if (x1 - x0 < 0.3) return;
    onWindow?.({ dmin: +x0.toFixed(3), dmax: +x1.toFixed(3),
                 yRange: view.ylog ? [10 ** lo, 10 ** hi] : [lo, hi] });
  };
  const onWheel = (e) => {
    e.preventDefault();
    const mult = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 120 : 1;
    const p = rel(e);
    const a = wheelAcc.current ??
      (wheelAcc.current = { dx: 0, dy: 0, pinch: 0, x: p.x, y: p.y });
    if (e.ctrlKey) a.pinch += e.deltaY * mult;
    else { a.dx += e.deltaX * mult; a.dy += e.deltaY * mult; }
    a.x = p.x; a.y = p.y;
    if (!a.raf) a.raf = requestAnimationFrame(flushWheel);
  };

  const panT = drag && mode === "pan"
    ? `translate(${drag.axis === "y" ? 0 : drag.x1 - drag.x0},${
                  drag.axis === "x" ? 0 : drag.y1 - drag.y0})` : null;
  const box = drag && mode === "zoom" ? {
    x: drag.axis === "y" ? m.l : Math.min(drag.x0, drag.x1),
    y: drag.axis === "x" ? m.t : Math.min(drag.y0, drag.y1),
    w: drag.axis === "y" ? pw : Math.abs(drag.x1 - drag.x0),
    h: drag.axis === "x" ? ph : Math.abs(drag.y1 - drag.y0),
  } : null;

  return html`
    <svg ref=${svgRef} width=${width} height=${height}
         viewBox="0 0 ${width} ${height}"
         class="chart mode-${mode}"
         onpointermove=${onMove} onpointerleave=${clearHover}
         onpointerdown=${onDown} onpointerup=${onUp}
         onwheel=${onWheel} ondblclick=${() => onHome?.()}>
      <defs>
        <clipPath id="plot"><rect x=${m.l} y=${m.t} width=${pw} height=${ph}/></clipPath>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="3.5" />
        </filter>
        ${shapes.map((s, i) => s.band && s.mi > 0 && html`
          <pattern id="hatch${i}" patternUnits="userSpaceOnUse"
                   width="5.5" height="5.5"
                   patternTransform="rotate(${s.mi === 1 ? 45 : 135})">
            <line x1="0" y1="0" x2="0" y2="5.5" stroke=${s.color}
                  stroke-width="1.3"/>
          </pattern>`)}
      </defs>

      <!-- grid -->
      ${xt.ticks.map((d) => html`
        <line x1=${X(10 ** d)} x2=${X(10 ** d)} y1=${m.t} y2=${bottomY}
              stroke=${theme.hair} stroke-width="1"/>`)}
      ${yAxis.ticks.map((t) => html`
        <line x1=${m.l} x2=${m.l + pw} y1=${yTickY(t)} y2=${yTickY(t)}
              stroke=${theme.hair} stroke-width="1"/>`)}

      <!-- data -->
      <g clip-path="url(#plot)">
        <g transform=${panT}>
          ${shapes.map((s, i) => s.band && html`
            <path d=${s.band}
                  fill=${s.mi > 0 ? `url(#hatch${i})` : s.color}
                  opacity=${s.mi > 0
                    ? Math.min(view.bandAlpha * 2.5, 0.8) : view.bandAlpha} />`)}
          ${shapes.map((s) => s.kind !== "element" && s.mi === 0 && html`
            <path d=${linePath(px, s.py)} fill="none" stroke=${s.color}
                  stroke-width="4" opacity=${theme.glow} filter="url(#glow)"/>`)}
          ${shapes.map((s) => html`
            <path d=${linePath(px, s.py)} fill="none" stroke=${s.color}
                  stroke-width=${s.kind === "total" ? (s.mi ? 2 : 2.6)
                                 : s.kind === "group" ? (s.mi ? 1.6 : 2) : 1.5}
                  stroke-dasharray=${s.kind === "element" ? "6 4" : MODEL_DASH[s.mi]}
                  stroke-linejoin="round" stroke-linecap="round"/>`)}
        </g>
        ${hover != null && !drag && hoverEnabled && html`
          <line x1=${px[hover]} x2=${px[hover]} y1=${m.t} y2=${bottomY}
                stroke=${theme.hair2} stroke-width="1"/>
          ${shapes.map((s) => Number.isFinite(s.py[hover]) && html`
            <circle cx=${px[hover]} cy=${s.py[hover]} r="3.5"
                    fill=${s.color} stroke=${theme.surface} stroke-width="1.5"/>`)}`}
        ${box && html`
          <rect x=${box.x} y=${box.y} width=${box.w} height=${box.h}
                fill=${theme.accentDim} stroke=${theme.accent}
                stroke-width="1" stroke-dasharray="4 3"/>`}
      </g>

      <!-- axes: thin frame lines + inward major/minor ticks -->
      <line x1=${m.l} x2=${m.l + pw} y1=${bottomY} y2=${bottomY}
            stroke=${theme.hair2} stroke-width="1"/>
      <line x1=${m.l} x2=${m.l} y1=${m.t} y2=${bottomY}
            stroke=${theme.hair2} stroke-width="1"/>
      ${xt.ticks.map((d) => html`
        <line x1=${X(10 ** d)} x2=${X(10 ** d)} y1=${bottomY} y2=${bottomY - 6}
              stroke=${theme.hair2} stroke-width="1"/>`)}
      ${xt.minors.map((v) => html`
        <line x1=${X(10 ** v)} x2=${X(10 ** v)} y1=${bottomY} y2=${bottomY - 3.5}
              stroke=${theme.hair2} stroke-width="1"/>`)}
      ${yAxis.ticks.map((t) => html`
        <line x1=${m.l} x2=${m.l + 6} y1=${yTickY(t)} y2=${yTickY(t)}
              stroke=${theme.hair2} stroke-width="1"/>`)}
      ${yAxis.minors.map((v) => html`
        <line x1=${m.l} x2=${m.l + 3.5} y1=${yTickY(v)} y2=${yTickY(v)}
              stroke=${theme.hair2} stroke-width="1"/>`)}

      <!-- tick labels (minor labels appear on deep zoom) -->
      ${xt.ticks.map((d) => html`
        <text x=${X(10 ** d)} y=${bottomY + 22} text-anchor="middle"
              class="tick">10${sup(d)}</text>`)}
      ${xt.minorLabels.map((L) => html`
        <text x=${X(10 ** L.v)} y=${bottomY + 22} text-anchor="middle"
              class="tickminor">${L.text}</text>`)}
      ${yAxis.ticks.map((t) => html`
        <text x=${m.l - 10} y=${yTickY(t) + 4} text-anchor="end" class="tick">
          ${view.ylog ? `10${sup(t)}` : sciLabel(t)}</text>`)}
      ${yAxis.minorLabels.map((L) => html`
        <text x=${m.l - 10} y=${yTickY(L.v) + 3.5} text-anchor="end"
              class="tickminor">${L.text}</text>`)}

      <!-- legend -->
      <g>
        <rect x=${lgX} y=${lgY} width=${lgW} height=${lgH} rx="7"
              fill=${theme.surface} fill-opacity="0.8"
              stroke=${theme.hair2} stroke-width="1"/>
        ${(() => {
          let y = lgY + LG.pad + 6;
          return legendRows.map((r) => {
            if (!r) { y += 8; return html`
              <line x1=${lgX + LG.pad} x2=${lgX + lgW - LG.pad}
                    y1=${y - LG.rowH / 2 - 1} y2=${y - LG.rowH / 2 - 1}
                    stroke=${theme.hair} stroke-width="1"/>`; }
            const row = html`
              <line x1=${lgX + LG.pad} x2=${lgX + LG.pad + LG.sample}
                    y1=${y - 4} y2=${y - 4} stroke=${r.color}
                    stroke-width=${r.lw} stroke-dasharray=${r.dash}
                    stroke-linecap="round"/>
              <text x=${lgX + LG.pad + LG.sample + LG.gap} y=${y}
                    class="serieslabel">${r.text}</text>`;
            y += LG.rowH;
            return row;
          });
        })()}
      </g>

      <!-- axis titles -->
      <text x=${m.l + pw / 2} y=${height - 14} text-anchor="middle"
            class="axistitle">${view.xTitle}</text>
      <text transform="translate(${m.l - 52},${m.t + ph / 2}) rotate(-90)"
            text-anchor="middle" class="axistitle">${view.yTitle}</text>
    </svg>`;
}

/* Axis-title strings (unicode math, STIX via CSS class) */
const AXSYMS = { etot: "E", ekin: "Eₖᵢₙ", rig: "R", en: "E_N", ekn: "Eₖᵢₙ,N" };

export function xTitle(basisMeta, basisKey) {
  const phrase = basisMeta.phrase[0].toUpperCase() + basisMeta.phrase.slice(1);
  return `${phrase}  ${AXSYMS[basisKey] ?? ""}  [${basisMeta.unit}]`;
}

export function yTitle(basisMeta, basisKey, gamma) {
  const u = basisMeta.unit;
  const g = +gamma.toFixed(2);
  if (g === 0) return `Φ  [(${u} m² s sr)⁻¹]`;
  return `${AXSYMS[basisKey]}${sup(g)} Φ  [${u}${sup(+(g - 1).toFixed(2))} m⁻² s⁻¹ sr⁻¹]`;
}
