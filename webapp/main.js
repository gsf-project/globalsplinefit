/* GSF Explorer — application shell (Preact + htm, no build step).
 *
 * State model: "params" require a Python round-trip (web worker, debounced);
 * "view" is applied client-side by chart.js on every render, so gamma,
 * scale, visibility, bands, pan and zoom respond instantly.
 */

import { html, render, useState, useEffect, useRef, useMemo }
  from "https://cdn.jsdelivr.net/npm/htm@3.1.1/preact/standalone.module.js";
import { Chart, THEMES, GROUPS, MODEL_DASH, X_DOMAIN, SAMPLE_TRIALS,
         sciLabel, sup, xTitle, yTitle } from "./chart.js";

const WHEEL = "./globalsplinefit-2.0.1-py3-none-any.whl";
const MAX_MODELS = 3;

/* ------------------------------------------------------------ worker rpc */

/* classic worker on purpose — module workers break older Firefox */
const worker = new Worker(new URL("./worker.js", import.meta.url));
let seq = 0;
const pending = new Map();
const listeners = { progress: () => {} };
worker.onmessage = ({ data }) => {
  if (data.type === "progress") { listeners.progress(data.stage); return; }
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  data.ok ? p.res(data.data) : p.rej(new Error(data.error));
};
const rpc = (cmd, args) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  worker.postMessage({ id, cmd, args });
});

/* -------------------------------------------------------------- helpers */

function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

const b64blob = (b64, mime) => {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Blob([buf], { type: mime });
};

const ymInt = (s) => parseInt(s.replace("-", ""), 10);
const exponential = (value, digits) =>
  Number.isFinite(value) ? value.toExponential(digits) : "—";

/* telegraphic state line shown next to the wordmark:
   MODEL(s) · quantity · modulation · range · abscissa */
function stateLine(p, unit, win, quantityLabel) {
  const mod = p.mod === "LIS" ? "LIS"
    : Array.isArray(p.mod)
      ? `${String(p.mod[0]).replace(/(\d{4})(\d{2})/, "$1/$2")}–${String(p.mod[1]).replace(/(\d{4})(\d{2})/, "$1/$2")}`
      : "SC24";
  const ABBR = { etot: "E / nucleus", ekin: "Eₖ / nucleus", rig: "rigidity",
                 en: "E / nucleon", ekn: "Eₖ / nucleon" };
  const dec = (d) => sup(Math.round(d * 10) / 10);
  return `${p.versions.join(" vs ")} · ${quantityLabel} · ${mod} · ` +
         `10${dec(win[0])}–10${dec(win[1])} ${unit} · ${ABBR[p.basis]}`;
}

/* Citation metadata lives in citations.json — the single source shared with
 * the documentation (mkdocs_hooks/citations.py renders the same file). Loaded
 * here with a top-level await so CITATIONS is ready before the first render. */
const CITATIONS = await fetch(new URL("./citations.json", import.meta.url))
  .then((r) => r.json())
  .then((d) => d.entries);

function CiteBlock({ c }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(c.bibtex); }
    catch {
      const ta = document.createElement("textarea");
      ta.value = c.bibtex;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };
  return html`
    <div class="cite">
      <div class="citehead">
        <strong>${c.label}</strong>
        ${c.url ? html`<a href=${c.url} target="_blank" rel="noopener">${c.key}</a>`
                : html`<span class="mtag">${c.key}</span>`}
        ${c.bibtex && html`
          <button class="action" onclick=${copy}>
            ${copied ? "copied ✓" : "Copy BibTeX"}</button>`}
      </div>
      ${c.bibtex ? html`<pre>${c.bibtex}</pre>`
                 : html`<p style="margin:4px 0 0">${c.note}</p>`}
    </div>`;
}

/* theme: explicit choice persists; otherwise follow the browser */
const prefersLight = () =>
  window.matchMedia?.("(prefers-color-scheme: light)").matches;
const initialTheme = () =>
  localStorage.getItem("gsfTheme") ?? (prefersLight() ? "light" : "dark");

/* ------------------------------------------------------------ components */

function Panel({ title, children, open = false }) {
  const [on, setOn] = useState(open);
  return html`
    <section class="panel ${on ? "" : "closed"}">
      <header onclick=${() => setOn(!on)} role="button"
              aria-expanded=${on} tabindex="0"
              onkeydown=${(e) => (e.key === "Enter" || e.key === " ") && setOn(!on)}>
        <h2>${title}</h2>
        <span class="chev">${on ? "−" : "+"}</span>
      </header>
      <div class="body">${children}</div>
    </section>`;
}

function ScrollBody({ label, children }) {
  const scrollerRef = useRef(null);
  const contentRef = useRef(null);
  const [metrics, setMetrics] = useState({ top: 0, client: 1, total: 1 });
  const measure = () => {
    const el = scrollerRef.current;
    if (!el) return;
    setMetrics({ top: el.scrollTop, client: el.clientHeight,
                 total: el.scrollHeight });
  };
  useEffect(() => {
    measure();
    const ro = new ResizeObserver(measure);
    if (contentRef.current) ro.observe(contentRef.current);
    return () => ro.disconnect();
  }, []);
  const overflow = metrics.total > metrics.client + 1;
  const thumbH = overflow
    ? Math.max(14, (metrics.client / metrics.total) * 100) : 100;
  const thumbTop = overflow
    ? (metrics.top / (metrics.total - metrics.client)) * (100 - thumbH) : 0;
  return html`
    <div class="panebody">
      <div class="panescroll" ref=${scrollerRef} tabindex="0"
           aria-label=${`${label} pane contents`} onscroll=${measure}>
        <div class="scrollcontent" ref=${contentRef}>${children}</div>
      </div>
      <div class="scrollrail ${overflow ? "active" : ""}" aria-hidden="true">
        <span style="height:${thumbH}%;top:${thumbTop}%"></span>
      </div>
    </div>`;
}

function Slider({ label, min, max, step, value, onInput, fmt = (v) => v }) {
  const pct = ((value - min) / (max - min)) * 100;
  return html`
    <label class="field">
      <span>${label}</span>
      <div class="sliderrow">
        <input type="range" min=${min} max=${max} step=${step} value=${value}
               style="--fill:${pct}%"
               oninput=${(e) => onInput(+e.target.value)} />
        <span class="val">${fmt(value)}</span>
      </div>
    </label>`;
}

function Seg({ options, value, onSelect }) {
  return html`
    <div class="seg" role="group">
      ${options.map((o) => html`
        <button class=${o.v === value ? "on" : ""}
                onclick=${() => onSelect(o.v)}>${o.label}</button>`)}
    </div>`;
}

const DashSample = ({ i, ink }) => html`
  <svg width="26" height="6" style="flex:none">
    <line x1="1" y1="3" x2="25" y2="3" stroke=${ink} stroke-width="2"
          stroke-dasharray=${MODEL_DASH[i]} stroke-linecap="round"/>
  </svg>`;

/* toolbar icons (inline, stroke = currentColor) */
const ICONS = {
  zoom: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6">
    <rect x="3" y="3" width="10" height="10" rx="1.5" stroke-dasharray="3 2.2"/>
    <line x1="13.5" y1="13.5" x2="18" y2="18" stroke-linecap="round"/></svg>`,
  pan: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 2v16M2 10h16M10 2l-2.4 2.4M10 2l2.4 2.4M10 18l-2.4-2.4M10 18l2.4-2.4M2 10l2.4-2.4M2 10l2.4 2.4M18 10l-2.4-2.4M18 10l-2.4 2.4"/></svg>`,
  home: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 9.5 10 3l7 6.5M5 8.5V17h10V8.5"/></svg>`,
  crosshair: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
    <circle cx="10" cy="10" r="4.5"/>
    <path d="M10 1.5v3M10 15.5v3M1.5 10h3M15.5 10h3"/></svg>`,
  sun: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round">
    <circle cx="10" cy="10" r="4"/>
    <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M4 4l1.4 1.4M14.6 14.6 16 16M16 4l-1.4 1.4M5.4 14.6 4 16"/></svg>`,
  moon: html`<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round">
    <path d="M16.5 12.5A7 7 0 0 1 7.5 3.5a7 7 0 1 0 9 9Z"/></svg>`,
};

/* ------------------------------------------------------------------- app */

const BASIS_LABELS = {
  etot: "Total energy / nucleus [GeV]",
  ekin: "Kinetic energy / nucleus [GeV]",
  rig: "Rigidity [GV]",
  en: "Total energy / nucleon [GeV]",
  ekn: "Kinetic energy / nucleon [GeV]",
};

function App() {
  const [stage, setStage] = useState("Starting");
  const [meta, setMeta] = useState(null);
  /* model state: requires Python. The grid always covers the full
     10^0–10^11 range so pan/zoom are pure display operations. */
  const [params, setParams] = useState({
    /* unrevisioned alias: the worker resolves it to the newest registered
       revision, so a model patch needs no change here. Replaced by
       meta.default once the worker answers. */
    versions: ["2026"], quantity: "nucleus", basis: "etot", npts: 480,
    elements: [], mod: "SC24", cutoff: 0, escale: 1.0, phiBins: 12,
    samples: 0,   // pseudo-experiment trials; 0 = error band
  });
  const [view, setView] = useState({
    gamma: 2.7, ylog: true, ratio: false, showTotal: true, showBands: true,
    bandAlpha: 0.16, lineWeight: 1, yRange: null, xWindow: [...X_DOMAIN],
    overlayBands: false,
    visible: { "H*": true, "He*": true, "O*": true, "Fe*": true },
  });
  const [hoverOn, setHoverOn] = useState(true);
  const [dragIdx, setDragIdx] = useState(null);
  const [overIdx, setOverIdx] = useState(null);
  const [months, setMonths] = useState(["2009-01", "2019-12"]);
  const [models, setModels] = useState(null);
  const [busy, setBusy] = useState(true);
  const [evalMs, setEvalMs] = useState(null);
  const [error, setError] = useState(null);
  const [hover, setHover] = useState(null);
  const [modal, setModal] = useState(null);
  const [about, setAbout] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [mode, setMode] = useState("zoom");
  const [overlay, setOverlay] = useState(null);
  const [theme, setTheme] = useState(initialTheme);
  const [exp, setExp] = useState({ preset: "full", fmt: "pdf", dpi: 300,
                                   csvN: 100, cov: "total", includeCaption: true });
  const [figureCaption, setFigureCaption] = useState("");

  listeners.progress = setStage;

  /* theme -> DOM + persistence; follow browser only while unchosen */
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: light)");
    if (!mq) return;
    const fn = (e) => {
      if (!localStorage.getItem("gsfTheme")) setTheme(e.matches ? "light" : "dark");
    };
    mq.addEventListener("change", fn);
    return () => mq.removeEventListener("change", fn);
  }, []);
  const pickTheme = (t) => { localStorage.setItem("gsfTheme", t); setTheme(t); };

  /* boot once — the default model comes from the package, not from here,
     so version renames/additions in globalsplinefit need no app change */
  useEffect(() => {
    (async () => {
      try {
        await rpc("boot", { wheelUrl: WHEEL });
        const m = await rpc("meta", { basis: "etot", version: null });
        setParams((p) => ({ ...p, versions: [m.default] }));
        setMeta(m);
      } catch (e) { setError(`Boot failed: ${e.message}`); }
    })();
  }, []);

  /* evaluate all active models on params change — debounced, and pumped
     sequentially so rapid input (e.g. scrolling the month picker) never
     queues stale recomputes: only the latest params are evaluated. */
  const evalParams = (version, withElements, p = params) => ({
    version, quantity: p.quantity, basis: p.basis,
    dmin: X_DOMAIN[0], dmax: X_DOMAIN[1],
    npts: p.npts, groups: GROUPS,
    elements: withElements ? p.elements : [],
    mod: p.mod, cutoff: p.cutoff, escale: p.escale, phiBins: p.phiBins,
    /* pseudo-experiments: primary model only (like the bands); the seed is
       fixed so a recompute redraws the same experiments */
    samples: withElements ? p.samples : 0, sampleSeed: 0,
  });
  const paramsKey = JSON.stringify(params);
  const latestKey = useRef(null);
  const pumping = useRef(false);
  const pump = async () => {
    if (pumping.current) return;
    pumping.current = true;
    let done = null;
    while (latestKey.current !== done) {
      const key = latestKey.current;
      const p = JSON.parse(key);
      try {
        const t0 = performance.now();
        const results = [];
        for (const [i, v] of p.versions.entries())
          results.push({ version: v,
                         data: await rpc("evaluate",
                                         { params: evalParams(v, i === 0, p) }) });
        if (latestKey.current === key) {
          setEvalMs(Math.round(performance.now() - t0));
          setModels(results);
          setError(null);
          setView((vw) => {
            const vis = { ...vw.visible };
            for (const s of results[0].data.series) vis[s.name] ??= true;
            return { ...vw, visible: vis };
          });
        }
      } catch (e) { if (latestKey.current === key) setError(e.message); }
      done = key;
    }
    pumping.current = false;
    setBusy(false);
  };
  useEffect(() => {
    if (!meta) return;
    setBusy(true);
    const t = setTimeout(() => { latestKey.current = paramsKey; pump(); },
                         models ? 220 : 0);
    return () => clearTimeout(t);
  }, [paramsKey, meta]);


  /* element list follows the primary model (e.g. deuterium exists only in
     isotope-format sets like 2026) */
  useEffect(() => {
    if (!meta) return;
    rpc("meta", { basis: params.basis, version: params.versions[0] })
      .then((m2) => {
        setMeta(m2);
        const avail = new Set(m2.elements.map((el) => el.z));
        setParams((p) => ({
          ...p, elements: p.elements.filter((z) => avail.has(z)) }));
      })
      .catch(() => {});
  }, [params.versions[0], params.basis]);

  /* stage size */
  const wrapRef = useRef(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const ro = new ResizeObserver(([e]) =>
      setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);
  const wide = size.w > 980;

  /* The dock's height varies: it reflows to two, three or four rows with
     the viewport and grows one when the sampling or comparison controls
     appear, so the chart's bottom inset is measured from it. The dock is
     laid out independently of the chart, so this cannot loop. */
  const dockRef = useRef(null);
  const [dockH, setDockH] = useState(0);
  useEffect(() => {
    /* border box: the dock's own padding is part of what the chart clears */
    const ro = new ResizeObserver(([e]) =>
      setDockH(e.target.getBoundingClientRect().height));
    ro.observe(dockRef.current);
    return () => ro.disconnect();
  }, []);

  /* These insets are stable while overlays open and close: the chart never
     resizes or recomputes merely because a command surface is visible. */
  const insets = wide
    ? { l: 14, r: 18, t: 72, b: Math.max(82, dockH + 14) }
    : { l: 0, r: 2, t: 126, b: Math.max(230, dockH + 14) };

  /* Command surfaces close consistently with Escape or an outside press.
     Their DOM is overlaid, never inserted into the chart's sizing flow. */
  useEffect(() => {
    if (!overlay) return;
    const onKey = (e) => {
      if (e.key === "Escape") setOverlay(null);
    };
    const onPointer = (e) => {
      if (!e.target.closest(".overlay-surface, .commandbtn")) setOverlay(null);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [overlay]);

  const setP = (patch) => setParams((p) => ({ ...p, ...patch }));
  const setV = (patch) => setView((v) => ({ ...v, ...patch }));
  const toggleSeries = (name) =>
    setView((v) => ({ ...v,
      visible: { ...v.visible, [name]: !v.visible[name] } }));
  const toggleElement = (z) => {
    const zn = (v) => (v === "D" ? 1.5 : v);   // D sorts right after H
    const on = params.elements.includes(z);
    setP({ elements: on ? params.elements.filter((e) => e !== z)
                        : [...params.elements, z].sort((a, b) => zn(a) - zn(b)) });
  };
  const addModel = (v) =>
    v && setP({ versions: [...params.versions, v] });
  const removeModel = (v) =>
    params.versions.length > 1 &&
    setP({ versions: params.versions.filter((x) => x !== v) });
  const moveModel = (from, to) => {
    if (from == null || to == null || from === to) return;
    const vs = [...params.versions];
    vs.splice(to, 0, vs.splice(from, 1)[0]);
    setP({ versions: vs });
  };
  const setPlotQuantity = (quantity) => {
    const validBases = meta?.quantities?.[quantity]?.bases ?? Object.keys(BASIS_LABELS);
    const basis = validBases.includes(params.basis) ? params.basis : validBases[0];
    setP({ quantity, basis });
    if (meta?.quantities?.[quantity]?.kind === "composition")
      setV({ ratio: false, ylog: false, yRange: null });
    else
      setV({ yRange: null });
  };

  /* navigation is view-only — no model recompute, the grid is cached */
  const goHome = () => setV({ xWindow: [...X_DOMAIN], yRange: null });
  const isHome = view.xWindow[0] === X_DOMAIN[0]
    && view.xWindow[1] === X_DOMAIN[1] && !view.yRange;
  const onWindow = (w) =>
    setV({ xWindow: [w.dmin, w.dmax], yRange: w.yRange });

  const basisMeta = meta?.bases[params.basis];
  const quantityMeta = meta?.quantities?.[params.quantity];
  const isComposition = quantityMeta?.kind === "composition";
  const validBases = quantityMeta?.bases ?? Object.keys(BASIS_LABELS);

  /* Exact fraction bands, in the background.
     sigma_i/Phi_total (what the chart draws immediately) treats the total as
     a constant although it contains Phi_i, so it overstates the band — by a
     few percent typically and by a lot where a group approaches the whole
     total. The covariance-aware fraction_error costs an order of magnitude
     more, which is seconds in WASM, so it is fetched after the plot is up
     and swapped in when it lands. The status pill says "refining bands…"
     meanwhile, so the band visibly tightening is explained rather than
     startling. */
  const [fracErr, setFracErr] = useState(null);
  const [refining, setRefining] = useState(false);
  const fracKey = view.ratio && view.showBands && !isComposition
    && !params.samples ? paramsKey : null;
  useEffect(() => {
    setFracErr(null);
    if (!fracKey || !models) { setRefining(false); return; }
    let cancelled = false;
    setRefining(true);
    rpc("fractionErrors", { params: evalParams(params.versions[0], true,
                                               JSON.parse(fracKey)) })
      .then((d) => { if (!cancelled) setFracErr(d.fractionErr); })
      .catch(() => {})
      .finally(() => { if (!cancelled) setRefining(false); });
    return () => { cancelled = true; };
  }, [fracKey, models]);
  const chartView = useMemo(() => (basisMeta ? {
    ...view,
    xTitle: xTitle(basisMeta, params.basis),
    yTitle: yTitle(basisMeta, params.basis, view.gamma, params.quantity,
                   view.ratio),
  } : view), [view, basisMeta, params.basis, params.quantity]);
  const T = THEMES[theme];

  /* Publication exports combine the active models on shared axes. */
  const PRESETS = {
    col1: { label: "Single column · 3.375×2.55 in", w: 3.375, h: 2.55 },
    col15: { label: "1.5 column · 5.0×3.55 in", w: 5.0, h: 3.55 },
    full: { label: "Paper wide · 6.0×3.6 in", w: 6.0, h: 3.6 },
  };
  const figureRequest = () => {
    const pr = PRESETS[exp.preset];
    return {
      params: params.versions.map((v, i) => ({
        ...evalParams(v, i === 0),
        dmin: view.xWindow[0], dmax: view.xWindow[1],
        groups: GROUPS.filter((g) => view.visible[g]),
      })),
      opts: { gamma: view.gamma, showTotal: view.showTotal,
              showBands: view.showBands, overlayBands: view.overlayBands,
              ratio: !isComposition && view.ratio,
              ylog: !isComposition && view.ylog, yRange: view.yRange,
              widthIn: pr.w, heightIn: pr.h, fmt: exp.fmt, dpi: exp.dpi,
              includeCaption: exp.includeCaption },
    };
  };
  const captionKey = overlay === "export" && !busy
    ? JSON.stringify(figureRequest()) : null;
  useEffect(() => {
    if (!captionKey) return;
    let cancelled = false;
    setFigureCaption("");
    rpc("figureCaption", JSON.parse(captionKey))
      .then((text) => { if (!cancelled) setFigureCaption(text); })
      .catch((e) => { if (!cancelled) setError(`Caption failed: ${e.message}`); });
    return () => { cancelled = true; };
  }, [captionKey]);
  const doFigure = async () => {
    setExporting(true);
    try {
      const mime = { pdf: "application/pdf", svg: "image/svg+xml",
                     png: "image/png" }[exp.fmt];
      const b64 = await rpc("figure", figureRequest());
      const tag = params.versions.map((v) => v.toLowerCase()).join("_vs_");
      download(`gsf_${tag}_${params.quantity}_${params.basis}.${exp.fmt}`,
               b64blob(b64, mime));
    } catch (e) { setError(`Export failed: ${e.message}`); }
    setExporting(false);
  };
  const doCsv = async () => {
    setExporting(true);
    try {
      for (const [i, v] of params.versions.entries()) {
        const text = await rpc("csv", {
          params: { ...evalParams(v, i === 0), npts: exp.csvN,
                    dmin: view.xWindow[0], dmax: view.xWindow[1],
                    groups: GROUPS.filter((g) => view.visible[g]) },
          covMode: isComposition ? "none" : exp.cov });
        download(`gsf_${v.toLowerCase()}_${params.quantity}_${params.basis}_${exp.csvN}pt.csv`,
                 new Blob([text], { type: "text/csv" }));
      }
    } catch (e) { setError(`CSV failed: ${e.message}`); }
    setExporting(false);
  };
  /* CSV of the data-table view: all models on the display grid */
  const doTableCsv = () => {
    if (!models) return;
    const head = [`x_${basisMeta?.unit ?? ""}`];
    const cols = [models[0].data.x];
    for (const mm of models) {
      const tag = models.length > 1 ? `_${mm.version}` : "";
      for (const s of mm.data.series) {
        const column = isComposition ? params.quantity
          : `${params.quantity}_flux_${s.name.replace("*", "star")}`;
        head.push(`${column}${tag}`);
        cols.push(s.flux);
        if (s.err) {
          head.push(`err_${column}${tag}`);
          cols.push(s.err);
        }
      }
      if (mm.data.total) {
        head.push(`${params.quantity}_flux_total${tag}`);
        cols.push(mm.data.total.flux);
        if (mm.data.total.err) {
          head.push(`err_total${tag}`);
          cols.push(mm.data.total.err);
        }
      }
    }
    const lines = [
      `# GSF Explorer display grid — ${params.versions.join(" vs ")} · ` +
        `${basisMeta?.phrase} [${basisMeta?.unit}] · ${models[0].data.modulation}`,
      head.join(","),
      ...models[0].data.x.map((_, i) =>
        cols.map((c) => Number.isFinite(c[i]) ? c[i].toExponential(6) : "")
          .join(",")),
    ];
    download(`gsf_view_${params.quantity}_${params.basis}_${params.npts}pt.csv`,
             new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  };
  const doSnapshot = () => {
    const svg = document.querySelector("svg.chart");
    if (!svg) return;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    const axisInk = getComputedStyle(svg).getPropertyValue("--ink-2").trim();
    style.textContent = `
      svg{background:${T.surface}}
      .tick,.tickminor{fill:${axisInk};font:13.5px 'IBM Plex Mono',monospace}
      .tickminor{font-size:11.5px}
      .serieslabel,.legendlabel,.legendtitle,.legendchev{fill:${T.ink2};font:500 12px 'IBM Plex Sans',sans-serif}
      .axistitle{fill:${T.ink2};font:500 13px 'IBM Plex Sans',sans-serif}`;
    clone.insertBefore(style, clone.firstChild);
    download("gsf_explorer_view.svg",
             new Blob([new XMLSerializer().serializeToString(clone)],
                      { type: "image/svg+xml" }));
  };
  const openAbout = async () => {
    setModal("about");
    if (!about) {
      try { setAbout(await rpc("about", { version: params.versions[0] })); }
      catch (e) { setAbout({ error: e.message }); }
    }
  };

  const modKind = params.mod === "LIS" ? "LIS"
    : Array.isArray(params.mod) ? "custom" : "SC24";
  /* the model requires start < end — sort, never error, on reversed input */
  const setInterval_ = (lo, hi) => {
    setMonths([lo, hi]);
    setP({ mod: [ymInt(lo), ymInt(hi)].sort((a, b) => a - b) });
  };
  const setMod = (kind) => {
    if (kind === "custom") setInterval_(months[0], months[1]);
    else setP({ mod: kind });
  };
  const y0 = meta?.phiYears?.[0] ?? 1975, y1 = meta?.phiYears?.[1] ?? 2025;
  const freeVersions =
    (meta?.versions ?? []).filter((v) => !params.versions.includes(v));
  const primary = models?.[0]?.data;
  const modelControls = html`
    <div class="serieslist">
      ${params.versions.map((v, i) => html`
        <div class="modelrow ${dragIdx === i ? "dragging" : ""} ${overIdx === i && dragIdx !== i ? "dragover" : ""}"
             draggable=${params.versions.length > 1}
             ondragstart=${(e) => { setDragIdx(i); e.dataTransfer.effectAllowed = "move"; }}
             ondragover=${(e) => { e.preventDefault(); setOverIdx(i); }}
             ondragleave=${() => setOverIdx(null)}
             ondrop=${(e) => { e.preventDefault(); moveModel(dragIdx, i); setDragIdx(null); setOverIdx(null); }}
             ondragend=${() => { setDragIdx(null); setOverIdx(null); }}>
          ${params.versions.length > 1 && html`<span class="grip" aria-hidden="true">⠿</span>`}
          <${DashSample} i=${i} ink=${T.ink} />
          <span class="mname">${v}</span>
          ${params.versions.length > 1 && html`
            <button class="mdel" title="Remove ${v}" aria-label=${`Remove ${v}`}
                    onclick=${() => removeModel(v)}>×</button>`}
        </div>`)}
    </div>
    ${params.versions.length > 1 && html`
      <p class="hinttext">Drag to reorder. The first model carries bands and elements.</p>`}
    ${params.versions.length < MAX_MODELS && freeVersions.length > 0 && html`
      <label class="field"><span>Add model (overlay)</span>
        <select value="" onchange=${(e) => { addModel(e.target.value); e.target.value = ""; }}>
          <option value="" disabled selected>Choose a parameter set…</option>
          ${freeVersions.map((v) => html`
            <option value=${v}>${v.endsWith("-USO") ? v.slice(0, -4) : v}${meta?.notes?.[v] ? `${meta.notes[v].startsWith("(") ? " " : " - "}${meta.notes[v]}` : ""}</option>`)}
        </select></label>`}
  `;

  return html`
    <div class="stage">
      <div class="chartwrap" ref=${wrapRef}>
        ${models && size.w > 0 && html`
          <${Chart} models=${models} view=${chartView} theme=${T} mode=${mode}
            xWindow=${view.xWindow} hoverEnabled=${hoverOn}
            width=${size.w} height=${size.h} insets=${insets}
            fractionErr=${fracErr}
            onWindow=${onWindow} onHome=${goHome} onHoverInfo=${setHover} />`}
        ${hover && html`
          <div class="hoverbox"
               style="left:${Math.max(12, Math.min(hover.pxX + 18, size.w - 224))}px;top:${insets.t + 42}px">
            <div class="hx">x = ${sciLabel(hover.x)} ${basisMeta?.unit ?? ""}</div>
            ${hover.rows.map((r) => html`
              <div class="hrow">
                <span class="swatch" style="background:${r.color}"></span>
                <span class="n">${r.name}${r.mi > 0 ? html`<span class="mtag"> ${r.tag}</span>` : ""}</span>
                <span class="v">${sciLabel(r.v)}</span>
              </div>`)}
          </div>`}
      </div>

      <header class="appheader">
        <div class="wordmark">
          <span class="name">GSF Explorer</span>
          <span class="tag">${meta
            ? stateLine(params, basisMeta?.unit ?? "GeV", view.xWindow,
                        quantityMeta?.label ?? params.quantity)
            : "global spline fit"}</span>
        </div>
        <div class="topright">
          <div class="statuspill ${busy || refining ? "busy" : ""}" role="status"
               aria-live="polite">
            <span class="dot"></span>
            <span>${busy ? (models ? "computing…" : stage)
              : refining ? "refining bands…"
              : `${params.versions[0]}${params.versions.length > 1 ? ` +${params.versions.length - 1}` : ""} · ${params.npts} pts · ${evalMs} ms`}</span>
          </div>
          <button class="iconbtn aboutbtn has-tooltip" title="About and citations"
                  aria-label="About and citations" onclick=${openAbout}>
            <span aria-hidden="true">i</span>
            <span class="tooltip">About and citations</span>
          </button>
          <button class="iconbtn has-tooltip" title="Switch theme"
                  aria-label=${`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
                  onclick=${() => pickTheme(theme === "dark" ? "light" : "dark")}>
            ${theme === "dark" ? ICONS.sun : ICONS.moon}
            <span class="tooltip">Switch theme</span>
          </button>
        </div>
      </header>

      ${models && html`
        <nav class="plottools" aria-label="Plot navigation">
          <button class="has-tooltip ${mode === "zoom" ? "on" : ""}"
                  title="Box zoom (drag a region)" aria-label="Box zoom"
                  aria-pressed=${mode === "zoom"} onclick=${() => setMode("zoom")}>
            ${ICONS.zoom}<span class="tooltip">Box zoom</span></button>
          <button class="has-tooltip ${mode === "pan" ? "on" : ""}"
                  title="Pan (drag to move the window)" aria-label="Pan"
                  aria-pressed=${mode === "pan"} onclick=${() => setMode("pan")}>
            ${ICONS.pan}<span class="tooltip">Pan</span></button>
          <button class="has-tooltip" title="Reset view (double-click plot does the same)"
                  aria-label="Reset view" disabled=${isHome} onclick=${goHome}>
            ${ICONS.home}<span class="tooltip">Reset view</span></button>
          <button class="has-tooltip ${hoverOn ? "on" : ""}"
                  title="Hover readout on/off" aria-label="Hover readout"
                  aria-pressed=${hoverOn}
                  onclick=${() => { setHoverOn(!hoverOn); setHover(null); }}>
            ${ICONS.crosshair}<span class="tooltip">Hover readout</span></button>
        </nav>`}

      <nav class="commandbar" aria-label="Explorer commands">
        ${[
          { id: "series", label: "Series" },
          { id: "settings", label: "Settings" },
          { id: "export", label: "Export" },
        ].map((command) => html`
          <button class="commandbtn ${overlay === command.id ? "on" : ""}"
                  aria-expanded=${overlay === command.id}
                  aria-controls=${`${command.id}-surface`}
                  onclick=${() => setOverlay(
                    overlay === command.id ? null : command.id)}>
            ${command.label}
          </button>`)}
      </nav>

      <section ref=${dockRef} class="displaydock" aria-label="Display controls">
        <div class="dockcontrol quantity-control">
          <span class="docklabel">Normalization</span>
          ${isComposition
            ? html`<span class="docknote">Moment</span>`
            : html`<${Seg} value=${view.ratio}
                onSelect=${(v) => setV({ ratio: v, yRange: null })}
                options=${[{ v: false, label: "Flux" },
                           { v: true, label: "Fraction" }]} />`}
        </div>
        <div class="dockcontrol gamma-control ${view.ratio || isComposition ? "disabled" : ""}">
          <${Slider} label="Spectral weight γ" min="0" max="3.2"
            step="0.05" value=${view.gamma}
            onInput=${(v) => setV({ gamma: v, yRange: null })}
            fmt=${(v) => v.toFixed(2)} />
          ${(view.ratio || isComposition) && html`
            <span class="docknote">${isComposition ? "Not applied to moments" : "Cancels in fraction view"}</span>`}
        </div>
        <div class="dockcontrol scale-control">
          <span class="docklabel">${isComposition ? "Moment" : view.ratio ? "Fraction" : "Flux"} scale</span>
          ${isComposition
            ? html`<span class="docknote">Linear</span>`
            : html`<${Seg} value=${view.ylog}
                onSelect=${(v) => setV({ ylog: v, yRange: null })}
                options=${[{ v: true, label: "Log" }, { v: false, label: "Linear" }]} />`}
        </div>
        <div class="dockcontrol band-control">
          <label class="switchcheck">
            <input type="checkbox" checked=${view.showBands}
                   onchange=${() => {
                     const on = !view.showBands;
                     setV({ showBands: on });
                     if (!on) setParams((p) => ({ ...p, samples: 0 }));
                   }} />
            <span class="switchtrack" aria-hidden="true"></span>
            <span>Uncertainty</span>
          </label>
          ${view.showBands && html`<${Seg} value=${params.samples > 0}
            onSelect=${(v) => setParams((p) => ({
              ...p, samples: v ? SAMPLE_TRIALS.def : 0 }))}
            options=${[{ v: false, label: "Band" },
                       { v: true, label: "Samples" }]} />`}
          ${view.showBands && params.samples === 0
            && params.versions.length > 1 && html`
            <label class="switchcheck compact">
              <input type="checkbox" checked=${view.overlayBands}
                     onchange=${() => setV({ overlayBands: !view.overlayBands })} />
              <span class="switchtrack" aria-hidden="true"></span>
              <span>Compared</span>
            </label>`}
          ${view.showBands && view.ratio && !isComposition && !params.samples
            && html`
            <span class="docknote" title=${fracErr
              ? "Covariance-aware fraction_error, including the correlation "
                + "with the total."
              : "Approximate band (σᵢ/Φ_total): the correlation with the "
                + "total is neglected until the exact one arrives."}>
              ${fracErr ? "exact fraction band" : "σᵢ/Φ_total, refining…"}</span>`}
        </div>
        <div class="dockcontrol style-controls ${view.showBands ? "split" : ""}">
          <div class="lineweight-control">
            <${Slider} label="Line weight" min="0.6" max="2" step="0.05"
              value=${view.lineWeight}
              onInput=${(v) => setV({ lineWeight: v })}
              fmt=${(v) => `${v.toFixed(2)}×`} />
          </div>
          ${view.showBands && params.samples === 0 && html`<div class="opacity-control">
            <${Slider} label="Band opacity" min="0.05" max="0.6" step="0.01"
              value=${view.bandAlpha} onInput=${(v) => setV({ bandAlpha: v })}
              fmt=${(v) => v.toFixed(2)} />
          </div>`}
          ${view.showBands && params.samples > 0 && html`<div class="opacity-control">
            <${Slider} label="Trials" min=${SAMPLE_TRIALS.min}
              max=${SAMPLE_TRIALS.max} step=${SAMPLE_TRIALS.step}
              value=${params.samples}
              onInput=${(v) => setParams((p) => ({ ...p, samples: Math.round(v) }))}
              fmt=${(v) => `${Math.round(v)}`} />
          </div>`}
        </div>
      </section>

      ${overlay && html`<div class="sheet-scrim" aria-hidden="true"></div>`}

      ${overlay === "series" && html`
        <aside id="series-surface" class="overlay-surface commandpane series-popover rail left"
               role="dialog" aria-modal="false" aria-label="Series visibility">
          <div class="surfacehead">
            <div><h2>Series</h2><p>Choose the curves shown on the canvas.</p></div>
            <button class="closebtn" aria-label="Close Series"
                    onclick=${() => setOverlay(null)}>×</button>
          </div>
          <${ScrollBody} label="Series">
            <${Panel} title="Model" open=${true}>
              ${modelControls}
            <//>
            <${Panel} title="Components" open=${true}>
              ${isComposition ? html`
                <p class="hinttext">Composition moments include every nucleus in the selected model.</p>
              ` : html`
              <div class="serieslist">
                <button class="seriesrow ${view.showTotal ? "" : "off"}"
                        aria-pressed=${view.showTotal}
                        onclick=${() => setV({ showTotal: !view.showTotal })}>
                  <span class="swatch" style="--c:${T.ink}"></span>${
                    params.quantity === "nucleon" ? "all-nucleon" : "all-particle"}
                </button>
                ${GROUPS.map((g) => html`
                  <button class="seriesrow ${view.visible[g] ? "" : "off"}"
                          aria-pressed=${!!view.visible[g]}
                          onclick=${() => toggleSeries(g)}>
                    <span class="swatch" style="--c:${T.groups[g]}"></span>${g}
                  </button>`)}
              </div>
              <div class="elementfield">
                <div class="fieldhead">
                  <span>Individual elements (dashed${params.versions.length > 1 ? ", primary model" : ""})</span>
                  <button class="resetbtn" disabled=${params.elements.length === 0}
                          onclick=${() => setP({ elements: [] })}>
                    Reset
                  </button>
                </div>
                <div class="elgrid">
                  ${(meta?.elements ?? []).map((el) => html`
                    <button class=${params.elements.includes(el.z) ? "on" : ""}
                            aria-pressed=${params.elements.includes(el.z)}
                            title=${el.z === "D" ? "Deuterium (Z=1, A=2)" : `Atomic number ${el.z}`}
                            onclick=${() => toggleElement(el.z)}>${el.sym}</button>`)}
                </div>
              </div>
              `}
            <//>
          <//>
        </aside>`}

      ${overlay === "settings" && html`
        <aside id="settings-surface" class="overlay-surface commandpane settings-popover rail left"
               role="dialog" aria-modal="false" aria-label="Settings">
          <div class="surfacehead">
            <div><h2>Settings</h2><p>Scientific model settings.</p></div>
            <button class="closebtn" aria-label="Close Settings"
                    onclick=${() => setOverlay(null)}>×</button>
          </div>
          <${ScrollBody} label="Settings">
            <${Panel} title="Abscissa" open=${true}>
            <label class="field"><span>Plot</span>
              <select aria-label="Plot quantity" value=${params.quantity}
                      onchange=${(e) => setPlotQuantity(e.target.value)}>
                ${Object.entries(meta?.quantities ?? {}).map(([key, q]) => html`
                  <option value=${key}>${q.label}</option>`)}
              </select></label>
            <label class="field"><span>Horizontal axis</span>
              <select value=${params.basis}
                      onchange=${(e) => setP({ basis: e.target.value })}>
                ${Object.entries(BASIS_LABELS)
                  .filter(([k]) => validBases.includes(k))
                  .map(([k, l]) => html`
                  <option value=${k}>${l}</option>`)}
              </select></label>
            <//>
            <${Panel} title="Solar modulation" open=${true}>
            <${Seg} value=${modKind} onSelect=${setMod}
              options=${[{ v: "SC24", label: "SC24" },
                         { v: "LIS", label: "LIS" },
                         { v: "custom", label: "Interval" }]} />
            ${modKind === "custom" && html`
              <div class="monthgrid">
                <label class="field"><span>From</span>
                  <input type="month" value=${months[0]}
                         min="${y0}-01" max="${y1}-12"
                         onchange=${(e) => setInterval_(e.target.value, months[1])} /></label>
                <label class="field"><span>To</span>
                  <input type="month" value=${months[1]}
                         min="${y0}-01" max="${y1}-12"
                         onchange=${(e) => setInterval_(months[0], e.target.value)} /></label>
              </div>`}
            <p class="hinttext">Potential data cover ${y0}–${y1}.</p>
            <//>
            <${Panel} title="Advanced" open=${true}>
            <label class="field"><span>Energy-scale factor</span>
              <input type="number" min="0.80" max="1.20" step="0.01"
                     value=${params.escale}
                     onchange=${(e) => setP({ escale: +e.target.value })} /></label>
            <label class="field"><span>Rigidity cutoff [GV] · 0 = off</span>
              <input type="number" min="0" max="30" step="0.5"
                     value=${params.cutoff}
                     onchange=${(e) => setP({ cutoff: +e.target.value })} /></label>
            <label class="field"><span>Grid points (full range, cached)</span>
              <${Seg} value=${params.npts}
                onSelect=${(v) => setP({ npts: v })}
                options=${[{ v: 240, label: "240" }, { v: 480, label: "480" },
                           { v: 960, label: "960" }]} /></label>
            <label class="field"><span>Solar-cycle averaging bins</span>
              <${Seg} value=${params.phiBins}
                onSelect=${(v) => setP({ phiBins: v })}
                options=${[{ v: 1, label: "1" }, { v: 6, label: "6" },
                           { v: 12, label: "12" }, { v: "full", label: "full" }]} /></label>
            <p class="hinttext">
              The period average bins the monthly potential φ and averages the
              modulated flux over the bins. The mean φ is exact at any setting,
              so only the curvature of flux(φ) changes: at 1 GeV, 1 bin is off
              by 5.2% from the full monthly average, 6 by 0.5%, 12 by 0.1% —
              all below 0.3% above 10 GeV, and irrelevant for LIS. “full”
              averages all ~130 months and takes seconds per update.
            </p>
            <//>
          <//>
        </aside>`}

      ${overlay === "export" && html`
        <aside id="export-surface" class="overlay-surface commandpane export-popover rail left"
               role="dialog" aria-modal="false" aria-label="Export">
          <div class="surfacehead">
            <div><h2>Export</h2><p>Download the selected series or live view.</p></div>
            <button class="closebtn" aria-label="Close Export"
                    onclick=${() => setOverlay(null)}>×</button>
          </div>
          <${ScrollBody} label="Export">
            <${Panel} title="Export" open=${true}>
            <span class="subhead">Publication figure</span>
            <label class="field">
              <span>Figure size${params.versions.length > 1 ? " · combined comparison" : ""}</span>
              <select value=${exp.preset}
                      onchange=${(e) => setExp({ ...exp, preset: e.target.value })}>
                ${Object.entries(PRESETS).map(([k, p]) => html`
                  <option value=${k}>${p.label}</option>`)}
              </select></label>
            <div class="btnrow">
              <${Seg} value=${exp.fmt}
                onSelect=${(v) => setExp({ ...exp, fmt: v })}
                options=${["pdf", "svg", "png"].map((f) => ({ v: f, label: f }))} />
              <button class="action primary" disabled=${exporting || busy}
                      onclick=${doFigure}>
                ${exporting ? "Rendering…" : "Download figure"}</button>
            </div>
            <label class="switchcheck caption-toggle">
              <input type="checkbox" checked=${exp.includeCaption}
                     onchange=${() => setExp({ ...exp, includeCaption: !exp.includeCaption })} />
              <span class="switchtrack" aria-hidden="true"></span>
              <span>Include figure caption</span>
            </label>
            <label class="field">
              <span>Figure caption</span>
              <textarea class="figure-caption" readonly rows="6"
                        value=${figureCaption} aria-label="Figure caption" />
            </label>
            <button class="action" disabled=${!figureCaption || busy}
                    onclick=${() => navigator.clipboard.writeText(figureCaption)
                      .catch((e) => setError(`Copy failed: ${e.message}`))}>Copy caption</button>
            <p class="hinttext">Paper fonts, colors, line weights and legends. The caption adds space below the chosen figure size.</p>
            <span class="subhead">Data</span>
            <label class="field">
              <span>CSV, current window${params.versions.length > 1 ? " · one file per model" : ""}</span>
              <div class="btnrow csvrow">
                <input type="number" min="20" max="300" step="10" value=${exp.csvN}
                       onchange=${(e) => setExp({ ...exp, csvN: +e.target.value })} />
                <button class="action" disabled=${exporting || busy}
                        onclick=${doCsv}>CSV</button>
              </div></label>
            <label class="field">
              <span>Covariance blocks</span>
              <select class="covsel" disabled=${isComposition} value=${exp.cov}
                      title="Per series appends one N×N covariance matrix for the total and for each drawn group and element; cross-group also appends the 6 mass-group cross-covariance blocks (H*×He*, …, O*×Fe*; diagonal = equal-energy)"
                      onchange=${(e) => setExp({ ...exp, cov: e.target.value })}>
                <option value="none">None (σ columns only)</option>
                <option value="total">Total flux</option>
                <option value="series">Total + each drawn series</option>
                <option value="cross">+ cross-group group pairs</option>
              </select></label>
            <span class="subhead">Canvas snapshot</span>
            <div class="btnrow">
              <button class="action" onclick=${doSnapshot}>View → SVG</button>
              <button class="action" onclick=${() => setModal("table")}>Data table</button>
            </div>
            <//>
          <//>
        </aside>`}

      ${error && html`<div class="toast">${error}</div>`}

      ${modal === "table" && primary && html`
        <div class="modalback" onclick=${(e) => e.target.classList.contains("modalback") && setModal(null)}>
          <div class="modal">
            <header>
              <h3>Model grid — ${quantityMeta?.label} ± σ${isComposition ? "" : ` (raw flux, ${basisMeta?.unit ?? ""} m² s sr)⁻¹`}${params.versions.length > 1 ? ` — ${params.versions.join(" vs ")}` : ""}</h3>
              <button class="action" style="margin-left:auto;margin-right:14px"
                      onclick=${doTableCsv}>Download CSV</button>
              <button onclick=${() => setModal(null)}>✕</button></header>
            <div class="mbody">
              <table class="data">
                <thead><tr>
                  <th>x [${basisMeta?.unit}]</th>
                  ${models.map((mm) => {
                    const tag = models.length > 1 ? ` · ${mm.version}` : "";
                    return html`
                      ${mm.data.series.map((s) => html`
                        <th>${isComposition ? quantityMeta?.label : `Φ(${s.name})`}${tag}</th><th>σ</th>`)}
                      ${mm.data.total && html`<th>Φ(total)${tag}</th><th>σ</th>`}`;
                  })}
                </tr></thead>
                <tbody>
                  ${primary.x.map((xv, i) => html`<tr>
                    <td>${xv.toExponential(3)}</td>
                    ${models.map((mm) => html`
                      ${mm.data.series.map((s) => html`
                        <td>${exponential(s.flux[i], 3)}</td>
                        <td>${s.err ? exponential(s.err[i], 2) : "—"}</td>`)}
                      ${mm.data.total && html`
                        <td>${exponential(mm.data.total.flux[i], 3)}</td>
                        <td>${mm.data.total.err ? exponential(mm.data.total.err[i], 2) : "—"}</td>`}`)}
                  </tr>`)}
                </tbody>
              </table>
            </div>
          </div>
        </div>`}

      ${modal === "about" && html`
        <div class="modalback" onclick=${(e) => e.target.classList.contains("modalback") && setModal(null)}>
          <div class="modal" style="width:min(720px, calc(100vw - 48px))">
            <header><h3>About — ${params.versions[0]}</h3>
              <button onclick=${() => setModal(null)}>✕</button></header>
            <div class="mbody about">
              <p><strong>Global Spline Fit (GSF)</strong> — a data-driven
              parametrization of the cosmic-ray flux and mass composition from
              ~1 GeV to 10¹¹ GeV, fitted to direct and indirect measurements
              with per-experiment energy-scale offsets. Four leading mass
              groups (H*, He*, O*, Fe*) on cubic B-splines; sub-leading
              abundances fixed from low-energy data. Uncertainties are
              propagated from the full fit covariance.</p>
              <p>This page runs the <strong>globalsplinefit</strong> Python
              package unmodified in your browser (Pyodide/WebAssembly);
              nothing is sent to a server.</p>
              <h4>Citations</h4>
              ${CITATIONS.map((c) => html`<${CiteBlock} c=${c} />`)}
              <h4>Parameter-set details (${params.versions[0]})</h4>
              ${about && html`<pre>${JSON.stringify(about, null, 2)}</pre>`}
            </div>
          </div>
        </div>`}

      ${!models && html`
        <div id="splash">
          <div class="tag">global spline fit</div>
          <div class="name">GSF Explorer</div>
          <div class="bar"></div>
          <div class="stage">${error ?? stage}</div>
          <div class="hint">Loading the scientific Python runtime and the
            globalsplinefit package (~25 MB, cached after the first visit).</div>
        </div>`}
    </div>`;
}

render(html`<${App} />`, document.getElementById("app"));
