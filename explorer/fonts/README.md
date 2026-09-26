# Paper figure fonts

Nimbus Roman is the font of the main paper's figures. These unmodified TrueType files come from
[Artifex's URW base 35 fonts](https://github.com/ArtifexSoftware/urw-base35-fonts/tree/master/fonts).
`LICENSE` and `COPYING` contain the font license and document embedding exception.
The IBM Plex web fonts used by the Explorer UI are under the SIL Open Font
License 1.1 (`OFL-IBM-Plex.txt`).

The Explorer worker loads these files when matplotlib is first needed.
`gsf_explorer.py` registers them for both browser and desktop rendering.
The font files are source assets and belong in Git with the export code.
