/* GSF Explorer — application shell (Preact + htm, no build step).
 *
 * State model: "params" require a Python round-trip (web worker, debounced);
 * "view" is applied client-side by chart.js on every render, so gamma,
 * scale, visibility, bands, pan and zoom respond instantly.
 */

import { html, render, useState, useEffect, useRef, useMemo }
  from "https://cdn.jsdelivr.net/npm/htm@3.1.1/preact/standalone.module.js";
import { Chart, THEMES, GROUPS, MODEL_DASH, X_DOMAIN, sciLabel, sup,
         xTitle, yTitle } from "./chart.js";

const WHEEL = "./globalsplinefit-2.0.0a1-py3-none-any.whl";
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

/* telegraphic state line shown next to the wordmark:
   MODEL(s) · modulation · range · abscissa */
function stateLine(p, unit, win) {
  const mod = p.mod === "LIS" ? "LIS"
    : Array.isArray(p.mod)
      ? `${String(p.mod[0]).replace(/(\d{4})(\d{2})/, "$1/$2")}–${String(p.mod[1]).replace(/(\d{4})(\d{2})/, "$1/$2")}`
      : "SC24";
  const ABBR = { etot: "E / nucleus", ekin: "Eₖ / nucleus", rig: "rigidity",
                 en: "E / nucleon", ekn: "Eₖ / nucleon" };
  const dec = (d) => sup(Math.round(d * 10) / 10);
  return `${p.versions.join(" vs ")} · ${mod} · ` +
         `10${dec(win[0])}–10${dec(win[1])} ${unit} · ${ABBR[p.basis]}`;
}

/* InspireHEP records, verbatim (fetched 2026-07-30) */
const CITATIONS = [
  {
    label: "GSF 2017 / 2019", key: "Dembinski:2017zsh",
    url: "https://inspirehep.net/literature/1639832",
    bibtex: `@article{Dembinski:2017zsh,
    author = "Dembinski, Hans Peter and Engel, Ralph and Fedynitch, Anatoli and Gaisser, Thomas and Riehn, Felix and Stanev, Todor",
    title = "{Data-driven model of the cosmic-ray flux and mass composition from 10 GeV to $10^{11}$ GeV}",
    eprint = "1711.11432",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.22323/1.301.0533",
    journal = "PoS",
    volume = "ICRC2017",
    pages = "533",
    year = "2018"
}`,
  },
  {
    label: "GSF 2024/2025 — UHECR 2024", key: "Fujisue:2025wnp",
    url: "https://inspirehep.net/literature/2907063",
    bibtex: `@article{Fujisue:2025wnp,
    author = "Fujisue, Kozo and Dembinski, Hans and Engel, Ralph and Fedynitch, Anatoli",
    title = "{Global Spline Fit (GSF) 2024}",
    doi = "10.22323/1.484.0087",
    journal = "PoS",
    volume = "UHECR2024",
    pages = "087",
    year = "2025"
}`,
  },
  {
    label: "GSF 2025 — ICRC 2025", key: "Dembinski:2025nmp",
    url: "https://inspirehep.net/literature/3067434",
    bibtex: `@article{Dembinski:2025nmp,
    author = "Dembinski, Hans and Engel, Ralph Richard and Fedynitch, Anatoli and Fujisue, Kozo",
    title = "{Global Spline Fit GSF-2025 - An update of the data-driven model of the cosmic-ray flux and its mass composition}",
    doi = "10.22323/1.501.0248",
    journal = "PoS",
    volume = "ICRC2025",
    pages = "248",
    year = "2025"
}`,
  },
  {
    label: "GSF 2026", key: "TBD",
    bibtex: null, note: "Publication in preparation — citation to follow.",
  },
];

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
    versions: ["2026"], basis: "etot", npts: 480,
    elements: [], mod: "SC24", cutoff: 0, escale: 1.0,
  });
  const [view, setView] = useState({
    gamma: 2.7, ylog: true, ratio: false, showTotal: true, showBands: true,
    bandAlpha: 0.16, lineWeight: 1, yRange: null, xWindow: [...X_DOMAIN],
    overlayBands: false,
    visible: { "p": true, "He": true, "O*": true, "Fe*": true },
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
                                   csvN: 100, cov: true });

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
    version, basis: p.basis, dmin: X_DOMAIN[0], dmax: X_DOMAIN[1],
    npts: p.npts, groups: GROUPS,
    elements: withElements ? p.elements : [],
    mod: p.mod, cutoff: p.cutoff, escale: p.escale,
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
  /* These insets are stable while overlays open and close: the chart never
     resizes or recomputes merely because a command surface is visible. */
  const insets = wide ? { l: 14, r: 18, t: 72, b: 82 }
                      : { l: 0, r: 2, t: 126, b: 230 };

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

  /* navigation is view-only — no model recompute, the grid is cached */
  const goHome = () => setV({ xWindow: [...X_DOMAIN], yRange: null });
  const isHome = view.xWindow[0] === X_DOMAIN[0]
    && view.xWindow[1] === X_DOMAIN[1] && !view.yRange;
  const onWindow = (w) =>
    setV({ xWindow: [w.dmin, w.dmax], yRange: w.yRange });

  const basisMeta = meta?.bases[params.basis];
  const chartView = useMemo(() => (basisMeta ? {
    ...view,
    xTitle: xTitle(basisMeta, params.basis),
    yTitle: view.ratio ? "Φ / Φ(all-particle)"
                       : yTitle(basisMeta, params.basis, view.gamma),
  } : view), [view, basisMeta, params.basis]);
  const T = THEMES[theme];

  /* exports (primary model) */
  const PRESETS = {
    prx1: { label: "PRX single column · 3.375×2.55 in", w: 3.375, h: 2.55 },
    prx15: { label: "1.5 column · 5.0×3.55 in", w: 5.0, h: 3.55 },
    full: { label: "Full width · 7.0×4.5 in", w: 7.0, h: 4.5 },
  };
  /* exports cover every active model — one file per parameter set */
  const doFigure = async () => {
    setExporting(true);
    try {
      const pr = PRESETS[exp.preset];
      const mime = { pdf: "application/pdf", svg: "image/svg+xml",
                     png: "image/png" }[exp.fmt];
      for (const [i, v] of params.versions.entries()) {
        const b64 = await rpc("figure", {
          params: { ...evalParams(v, i === 0),
                    dmin: view.xWindow[0], dmax: view.xWindow[1],
                    groups: GROUPS.filter((g) => view.visible[g]) },
          opts: { gamma: view.gamma, showTotal: view.showTotal,
                  showBands: view.showBands,
                  bandAlpha: Math.max(view.bandAlpha, 0.12),
                  ylog: view.ylog, widthIn: pr.w, heightIn: pr.h,
                  fmt: exp.fmt, dpi: exp.dpi },
        });
        download(`gsf_${v.toLowerCase()}_${params.basis}.${exp.fmt}`,
                 b64blob(b64, mime));
      }
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
          includeCov: exp.cov });
        download(`gsf_${v.toLowerCase()}_${params.basis}_${exp.csvN}pt.csv`,
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
        head.push(`flux_${s.name.replace("*", "star")}${tag}`);
        cols.push(s.flux);
        if (s.err) {
          head.push(`err_${s.name.replace("*", "star")}${tag}`);
          cols.push(s.err);
        }
      }
      if (mm.data.total) {
        head.push(`flux_total${tag}`);
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
        cols.map((c) => c[i].toExponential(6)).join(",")),
    ];
    download(`gsf_view_${params.basis}_${params.npts}pt.csv`,
             new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  };
  const doSnapshot = () => {
    const svg = document.querySelector("svg.chart");
    if (!svg) return;
    const clone = svg.cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = `
      svg{background:${T.surface}}
      .tick{fill:${T.ink3};font:13.5px 'IBM Plex Mono',monospace}
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
            <option value=${v}>${v}${meta?.notes?.[v] ? ` — ${meta.notes[v]}` : ""}</option>`)}
        </select></label>`}
  `;

  return html`
    <div class="stage">
      <div class="chartwrap" ref=${wrapRef}>
        ${models && size.w > 0 && html`
          <${Chart} models=${models} view=${chartView} theme=${T} mode=${mode}
            xWindow=${view.xWindow} hoverEnabled=${hoverOn}
            width=${size.w} height=${size.h} insets=${insets}
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
            ? stateLine(params, basisMeta?.unit ?? "GeV", view.xWindow)
            : "global spline fit"}</span>
        </div>
        <div class="topright">
          <div class="statuspill ${busy ? "busy" : ""}" role="status" aria-live="polite">
            <span class="dot"></span>
            <span>${busy ? (models ? "computing…" : stage)
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

      <section class="displaydock" aria-label="Display controls">
        <div class="dockcontrol quantity-control">
          <span class="docklabel">Quantity</span>
          <${Seg} value=${view.ratio}
            onSelect=${(v) => setV({ ratio: v, yRange: null })}
            options=${[{ v: false, label: "Flux" },
                       { v: true, label: "Ratio" }]} />
        </div>
        <div class="dockcontrol gamma-control ${view.ratio ? "disabled" : ""}">
          <${Slider} label="Spectral weight γ" min="0" max="3.2"
            step="0.05" value=${view.gamma}
            onInput=${(v) => setV({ gamma: v, yRange: null })}
            fmt=${(v) => v.toFixed(2)} />
          ${view.ratio && html`<span class="docknote">Cancels in ratio view</span>`}
        </div>
        <div class="dockcontrol scale-control">
          <span class="docklabel">${view.ratio ? "Ratio" : "Flux"} scale</span>
          <${Seg} value=${view.ylog}
            onSelect=${(v) => setV({ ylog: v, yRange: null })}
            options=${[{ v: true, label: "Log" }, { v: false, label: "Linear" }]} />
        </div>
        <div class="dockcontrol band-control">
          <label class="switchcheck">
            <input type="checkbox" checked=${view.showBands}
                   onchange=${() => setV({ showBands: !view.showBands })} />
            <span class="switchtrack" aria-hidden="true"></span>
            <span>Bands</span>
          </label>
          ${view.showBands && params.versions.length > 1 && html`
            <label class="switchcheck compact">
              <input type="checkbox" checked=${view.overlayBands}
                     onchange=${() => setV({ overlayBands: !view.overlayBands })} />
              <span class="switchtrack" aria-hidden="true"></span>
              <span>Compared</span>
            </label>`}
        </div>
        <div class="dockcontrol style-controls ${view.showBands ? "split" : ""}">
          <div class="lineweight-control">
            <${Slider} label="Line weight" min="0.6" max="2" step="0.05"
              value=${view.lineWeight}
              onInput=${(v) => setV({ lineWeight: v })}
              fmt=${(v) => `${v.toFixed(2)}×`} />
          </div>
          ${view.showBands && html`<div class="opacity-control">
            <${Slider} label="Band opacity" min="0.05" max="0.6" step="0.01"
              value=${view.bandAlpha} onInput=${(v) => setV({ bandAlpha: v })}
              fmt=${(v) => v.toFixed(2)} />
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
              <div class="serieslist">
                <button class="seriesrow ${view.showTotal ? "" : "off"}"
                        aria-pressed=${view.showTotal}
                        onclick=${() => setV({ showTotal: !view.showTotal })}>
                  <span class="swatch" style="--c:${T.ink}"></span>all-particle
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
            <label class="field"><span>Horizontal axis</span>
              <select value=${params.basis}
                      onchange=${(e) => setP({ basis: e.target.value })}>
                ${Object.entries(BASIS_LABELS).map(([k, l]) => html`
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
            <//>
          <//>
        </aside>`}

      ${overlay === "export" && html`
        <aside id="export-surface" class="overlay-surface commandpane export-popover rail left"
               role="dialog" aria-modal="false" aria-label="Export">
          <div class="surfacehead">
            <div><h2>Export</h2><p>Download the current model or live view.</p></div>
            <button class="closebtn" aria-label="Close Export"
                    onclick=${() => setOverlay(null)}>×</button>
          </div>
          <${ScrollBody} label="Export">
            <${Panel} title="Export" open=${true}>
            <span class="subhead">Publication figure</span>
            <label class="field">
              <span>Paper style${params.versions.length > 1 ? " · one file per model" : ""}</span>
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
                ${exporting ? "Rendering…" : "Download"}</button>
            </div>
            <span class="subhead">Data</span>
            <label class="field">
              <span>CSV, current window${params.versions.length > 1 ? " · one file per model" : ""}</span>
              <div class="btnrow csvrow">
                <input type="number" min="20" max="300" step="10" value=${exp.csvN}
                       onchange=${(e) => setExp({ ...exp, csvN: +e.target.value })} />
                <label class="check">
                  <input type="checkbox" checked=${exp.cov}
                         onchange=${() => setExp({ ...exp, cov: !exp.cov })} />
                  Covariance</label>
                <button class="action" disabled=${exporting || busy}
                        onclick=${doCsv}>CSV</button>
              </div></label>
            <span class="subhead">Live view</span>
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
              <h3>Model grid — Φ ± σ (raw flux, ${basisMeta?.unit ?? ""} m² s sr)⁻¹${params.versions.length > 1 ? ` — ${params.versions.join(" vs ")}` : ""}</h3>
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
                        <th>Φ(${s.name})${tag}</th><th>σ</th>`)}
                      ${mm.data.total && html`<th>Φ(total)${tag}</th><th>σ</th>`}`;
                  })}
                </tr></thead>
                <tbody>
                  ${primary.x.map((xv, i) => html`<tr>
                    <td>${xv.toExponential(3)}</td>
                    ${models.map((mm) => html`
                      ${mm.data.series.map((s) => html`
                        <td>${s.flux[i].toExponential(3)}</td>
                        <td>${s.err ? s.err[i].toExponential(2) : "—"}</td>`)}
                      ${mm.data.total && html`
                        <td>${mm.data.total.flux[i].toExponential(3)}</td>
                        <td>${mm.data.total.err ? mm.data.total.err[i].toExponential(2) : "—"}</td>`}`)}
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
              groups (p, He, O*, Fe*) on cubic B-splines; sub-leading
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
