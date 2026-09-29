"""Render citation metadata into the docs from the single shared source.

`webapp/citations.json` is the one place citations are edited; the Explorer
fetches it at runtime and this hook renders it into the pages at build time,
so the two can never drift. Pages opt in with a placeholder comment:

    <!-- citations:recommended -->   the citation to use today, as BibTeX
    <!-- citations:all -->           every GSF citation, collapsible
    <!-- citations:potentials -->    the solar-modulation potential papers
    <!-- citations:versions -->      version -> citation mapping (table rows)
"""

import json
import pathlib

CITATIONS = pathlib.Path(__file__).resolve().parents[1] / "webapp" / "citations.json"


def _load():
    return json.loads(CITATIONS.read_text())


def _entry(e):
    return {c["key"]: c for c in e["entries"]}


def _source(cit):
    return "arXiv" if "arxiv.org" in cit["url"] else "InspireHEP"


def _bibtex_block(cit):
    if not cit.get("bibtex"):
        return f'!!! note "{cit["label"]}"\n\n    {cit.get("note", "")}\n'
    link = f" — [{_source(cit)}]({cit['url']})" if cit.get("url") else ""
    return (
        f'??? quote "{cit["label"]} — `{cit["key"]}`{link}"\n\n'
        + "    ```bibtex\n"
        + "".join(f"    {line}\n" for line in cit["bibtex"].splitlines())
        + "    ```\n"
    )


def _recommended(data):
    cits = _entry(data)
    cit = cits[data["recommended"]]
    out = [data.get("recommended_note", ""), "", "```bibtex", cit["bibtex"], "```"]
    if cit.get("url"):
        out += ["", f"{_source(cit)} record: [{cit['key']}]({cit['url']})"]
    return "\n".join(out)


def _all(data):
    return "\n".join(_bibtex_block(c) for c in data["entries"])


def _potentials(data):
    rows = ["| Potential | Reference | Used by |", "|---|---|---|"]
    used = {
        "Ghelfi:2016pcv": "`2026.1`, `2026.1-SIB23e`, `2026.1-EPOSLHCR`",
        "Usoskin:2017cli": "`2026.1-USO`, `2025`, `2019`, `2017`",
    }
    for p in data["solar_modulation_potentials"]:
        rows.append(
            f"| {p['label']} | [`{p['key']}`]({p['url']}) | {used.get(p['key'], '')} |"
        )
    rows += ["", ""]
    for p in data["solar_modulation_potentials"]:
        rows.append(f"- **{p['label']}** — {p['note']}")
    return "\n".join(rows)


def _versions(data):
    cits = _entry(data)
    rows = ["| Version | Cite |", "|---|---|"]
    for ver, keys in data["versions"].items():
        refs = []
        for k in keys:
            c = cits[k]
            refs.append(f"[`{c['key']}`]({c['url']})" if c.get("url") else c["label"])
        rows.append(f'| `"{ver}"` | {", ".join(refs)} |')
    return "\n".join(rows)


RENDERERS = {
    "recommended": _recommended,
    "all": _all,
    "potentials": _potentials,
    "versions": _versions,
}


def on_page_markdown(markdown, **kwargs):  # noqa: ARG001 — mkdocs hook signature
    if "<!-- citations:" not in markdown:
        return markdown
    data = _load()
    for name, render in RENDERERS.items():
        markdown = markdown.replace(f"<!-- citations:{name} -->", render(data))
    return markdown
