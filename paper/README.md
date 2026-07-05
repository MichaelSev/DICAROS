# Application note

`application_note.tex` is a self-contained Bioinformatics-style Application Note
(<= 4 pages, two-column, with the required *Availability and Implementation*
statement).

## Build

```bash
cd paper
pdflatex application_note
bibtex   application_note
pdflatex application_note
pdflatex application_note
```

Requires a TeX distribution (TeX Live / MiKTeX). No journal `.cls` is needed;
the body can be pasted into the official OUP/Bioinformatics template at
submission time.

## Figures

`figures/*_results.png` are **placeholders** until you generate the real
results. From the repository root:

```bash
OMP_NUM_THREADS=8 python ../scripts/make_results.py
```

This reconstructs both datasets and overwrites the figures with the real
phylomorphospace + shape-overlay panels, and writes `results_summary.csv` and
`results_table.tex`. The hard-coded counts in Table 1 of the note
(tips/ancestors/landmarks/singletons) already match the bundled datasets.
