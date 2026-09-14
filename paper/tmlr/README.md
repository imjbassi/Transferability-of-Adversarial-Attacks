# Anonymous TMLR submission build

This directory vendors the official TMLR style files retrieved from
`JmlrOrg/tmlr-style-file` on 2026-09-14. The style files are unmodified.

Build with:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

`main.tex` is the anonymous review version. It intentionally omits the author,
email address, repository commit identifier, GitHub URL, and Zenodo DOI. Use the
public manuscript in `paper/main.tex` for an attributed preprint or archival copy.
