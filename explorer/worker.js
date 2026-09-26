/* GSF Explorer — Pyodide web worker.
 *
 * Runs the real globalsplinefit package (numpy + scipy, WASM) off the main
 * thread. RPC: postMessage({id, cmd, args}) -> {id, ok, data | error}.
 * Boot progress is streamed as {type: "progress", stage}.
 * matplotlib is installed lazily on the first publication-figure export.
 *
 * Deliberately a CLASSIC worker (importScripts, no ES modules): module
 * workers are unsupported in older Firefox, and a silent failure here
 * blanks the whole app.
 */

importScripts("https://cdn.jsdelivr.net/pyodide/v0.28.3/full/pyodide.js");

let pyodide = null;
let haveMatplotlib = false;

const progress = (stage) => postMessage({ type: "progress", stage });

async function boot(wheelUrl) {
  progress("Loading Python runtime");
  pyodide = await loadPyodide();

  progress("Loading numpy + scipy");
  await pyodide.loadPackage(["micropip"]);
  const micropip = pyodide.pyimport("micropip");

  progress("Installing globalsplinefit");
  await micropip.install(wheelUrl);

  progress("Loading model core");
  for (const f of ["gsf_explorer.py", "bridge.py"]) {
    // no-cache: always revalidate so a reloaded page never runs stale model code
    const res = await fetch(new URL(`./${f}`, self.location.href),
                            { cache: "no-cache" });
    pyodide.FS.writeFile(f, await res.text());
  }
  pyodide.runPython("import bridge");

  progress("Evaluating model");
  return true;
}

async function ensureMatplotlib() {
  if (haveMatplotlib) return;
  progress("Loading matplotlib (first export only)");
  await pyodide.loadPackage(["matplotlib"]);
  progress("Loading paper fonts");
  pyodide.FS.mkdirTree("fonts");
  await Promise.all(["Regular", "Italic", "Bold", "BoldItalic"].map(async (variant) => {
    const name = `NimbusRoman-${variant}.ttf`;
    const res = await fetch(new URL(`./fonts/${name}`, self.location.href));
    if (!res.ok) throw new Error(`Could not load paper font ${variant}`);
    pyodide.FS.writeFile(`fonts/${name}`, new Uint8Array(await res.arrayBuffer()));
  }));
  haveMatplotlib = true;
}

const py = (expr, vars) => {
  const g = pyodide.toPy(vars || {});
  try {
    return pyodide.runPython(expr, { globals: g });
  } finally {
    g.destroy();
  }
};

const call = {
  boot: (a) => boot(a.wheelUrl),

  meta: (a) =>
    JSON.parse(py("import bridge; bridge.meta(basis, version)",
                  { basis: a.basis, version: a.version })),

  evaluate: (a) =>
    JSON.parse(py("import bridge; bridge.evaluate(params)",
                  { params: JSON.stringify(a.params) })),

  fractionErrors: (a) =>
    JSON.parse(py("import bridge; bridge.fraction_errors(params)",
                  { params: JSON.stringify(a.params) })),

  csv: (a) =>
    py("import bridge; bridge.csv(params, cov_mode)",
       { params: JSON.stringify(a.params), cov_mode: a.covMode || "none" }),

  figureCaption: (a) =>
    py("import bridge; bridge.figure_caption(params, opts)",
       { params: JSON.stringify(a.params), opts: JSON.stringify(a.opts) }),

  figure: async (a) => {
    await ensureMatplotlib();
    return py("import bridge; bridge.figure(params, opts)",
              { params: JSON.stringify(a.params),
                opts: JSON.stringify(a.opts) });
  },

  about: (a) =>
    JSON.parse(py("import bridge; bridge.about(version)",
                  { version: a.version })),
};

onmessage = async (ev) => {
  const { id, cmd, args } = ev.data;
  try {
    const data = await call[cmd](args || {});
    postMessage({ id, ok: true, data });
  } catch (err) {
    postMessage({ id, ok: false,
                  error: String((err && err.message) || err) });
  }
};
