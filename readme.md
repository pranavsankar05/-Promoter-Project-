# Statistical and Machine Learning-Based Identification of σ⁷⁰ Promoter Regions in *E. coli* K-12 Using DNA Sequence Motifs

Course project for **Biology for CSE (AI)**, 3rd semester, B.Tech Computer Science (Artificial Intelligence), Amrita Vishwa Vidyapeetham, Bengaluru.

---

## Table of contents

1. [Project summary](#1-project-summary)
2. [Background in simple words](#2-background-in-simple-words)
3. [Research question](#3-research-question)
4. [Main results and inference](#4-main-results)
5. [Repository structure](#5-repository-structure)
6. [Setup](#6-setup)
7. [How to run](#7-how-to-run)
8. [Pipeline in detail](#8-pipeline-in-detail)
9. [Data sources](#9-data-sources)
10. [File formats and column schemas](#10-file-formats-and-column-schemas)
11. [Design decisions and reasons](#11-design-decisions-and-reasons)
12. [Evaluation](#12-evaluation)
13. [Limitations](#13-limitations)
14. [Reproducibility](#14-reproducibility)
15. [Troubleshooting](#15-troubleshooting)
16. [Course mapping](#16-course-mapping)
17. [Team](#17-team)
18. [References](#18-references)

---

## 1. Project summary

We teach a computer to tell **σ⁷⁰ promoter DNA** from **non-promoter DNA** in *Escherichia coli* K-12 MG1655 (RefSeq NC_000913.3, 4,641,652 bp). We then test whether giving the model the two known promoter patterns (the **−35 box**, about `TTGACA`, and the **−10 box**, about `TATAAT`) helps more than giving it plain letter counts (how much A, T, G, C).

The project is three Python scripts, run in order:

| Script | What it does | Main output |
|---|---|---|
| `01_data_audit.py` | Downloads the genome, loads RegulonDB promoters, keeps strong σ⁷⁰ promoters, cuts verified 81 bp windows | `data/processed/positives.csv` |
| `02_build_background.py` | Samples GC-matched non-promoter windows from the genome | `data/processed/dataset_all.csv` |
| `03_model.py` | Splits by genome position, fits motif scorers, trains and compares models | `results/results_table.csv`, `results/pr_curve.png` |

---

## 2. Background in simple words

- **Promoter:** a stretch of DNA just before a gene where the RNA polymerase attaches to start reading the gene.
- **σ⁷⁰ (sigma-70):** the main "guide" protein (sigma factor) that helps RNA polymerase find promoters in *E. coli*. Promoters it recognises share two short patterns.
- **−35 box (about `TTGACA`) and −10 box (about `TATAAT`):** the two patterns, located roughly 35 and 10 letters before the transcription start site. Real promoters match them loosely, not exactly.
- **TSS (+1):** the transcription start site, the first DNA letter that is copied into RNA.
- **PWM (position weight matrix):** a table that scores how much a 6-letter stretch looks like a real box, letter by letter, relative to background DNA.
- **PR-AUC:** area under the precision-recall curve. It measures how well a model ranks real promoters above non-promoters without depending on a single cut-off. Random guessing scores the share of promoters in the test set.

---

## 3. Research question

> Does explicit −10/−35 motif modelling beat composition-only features at separating annotated σ⁷⁰ promoters from selected background regions in *E. coli* K-12 MG1655 (NC_000913.3)?

---

## 4. Main results and inference

All numbers are on the **held-out test set** (the last 20% of rows by genome position): 424 rows, 36.3% promoters (about 154 promoters).

### 4.1 Results table

| Model | Features | PR-AUC | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|
| Baseline | −10 PWM score + −35 PWM score | 0.6528 | 0.6275 | 0.6234 | 0.6254 | 5.6117 (best F1 on train) |
| A | Composition only | 0.5833 | 0.5029 | 0.5714 | 0.5350 | 0.5 |
| B | Composition + motif features | **0.7300** | 0.6292 | 0.7273 | **0.6747** | 0.5 |
| Random guess | n/a | 0.363 | n/a | n/a | n/a | n/a |

### 4.2 Figure

![Precision-recall curves on the held-out test set](results/pr_curve.png)

*Precision-recall curves for the baseline, model A and model B, with the random-guess line at 0.363.*

### 4.3 Inference

1. **Motif features help.** Adding −10/−35 motif scores to composition (A → B) raises PR-AUC from 0.583 to 0.730 (about +0.15) and F1 from 0.535 to 0.675. This answers the research question with "yes".
2. **Two motif scores beat 68 composition features.** The motif-only baseline (PR-AUC 0.653) beats composition alone (0.583). Because the background is GC-matched, plain letter counts carry little promoter signal, while the −10/−35 boxes carry a lot.
3. **Motif and composition are complementary.** B beats both the baseline and A, so composition adds information on top of the motifs.
4. **Recall improves most.** B finds about 73% of test promoters (recall 0.727) at precision about 0.63, so roughly 6 in 10 predicted promoters are real.
5. **All models beat random guessing** (0.363), so the signal is real. A is only a small step above guessing.
6. **Absolute scores are moderate on purpose.** The background is GC-matched real genome sequence, a hard control. The relative gap between models is the finding, not the absolute value.

### 4.4 Conclusion

Explicit σ⁷⁰ promoter motif modelling (−10 and −35 PWM scores) improves identification of annotated promoters over composition-only features in *E. coli* K-12, under a GC-matched background and a position-based train/test split.

### 4.5 How far to trust this

- Differences are **descriptive**. No significance tests or confidence intervals were run, and the test set has only about 154 promoters.
- Thresholds differ (0.5 for A and B, a train-chosen value for the baseline), so compare models mainly by PR-AUC.
- The background is "selected", not proven non-promoter, and windows are aligned at the TSS. See [Limitations](#13-limitations).

## 5. Repository structure

```
sigma70/
├── 01_data_audit.py
├── 02_build_background.py
├── 03_model.py
├── README.md
├── .gitignore                  (contains data/raw/)
├── data/
│   ├── raw/                    (not committed)
│   │   ├── PromoterSet.tsv     (downloaded by hand from RegulonDB)
│   │   └── NC_000913.3.fasta   (downloaded automatically by 01)
│   └── processed/
│       ├── positives.csv       (made by 01)
│       └── dataset_all.csv     (made by 02)
└── results/
    ├── audit_counts.csv        (made by 01)
    ├── results_table.csv       (made by 03)
    └── pr_curve.png            (made by 03)
```

`data/raw/` is listed in `.gitignore` because the files are large and can be re-downloaded.

---

## 6. Setup

**Tested on:** macOS, VS Code terminal, conda (base) environment. Python version and library versions were not pinned. Any recent Python 3 with the libraries below should work.

**Libraries**

```
pip install pandas numpy scikit-learn matplotlib
```

- `pandas`, `numpy`: data handling
- `scikit-learn`: logistic regression, scaling, metrics
- `matplotlib`: the precision-recall figure (uses the non-interactive `Agg` backend, so no window opens)

**Internet:** needed once, so `01_data_audit.py` can download the genome from NCBI.

**Promoter file (manual step)**

1. Open RegulonDB (regulondb.ccg.unam.mx) and go to the downloads section.
2. Download the dataset **PromoterSet** ("Promoters information and their first transcribed genes").
3. Save it as `data/raw/PromoterSet.tsv`.

**Git**

Add this line to `.gitignore`:

```
data/raw/
```

---

## 7. How to run

Open a terminal **inside the project folder** (check with `pwd`) and run the scripts in order:

```
python 01_data_audit.py
python 02_build_background.py
python 03_model.py
```

Run order matters: 02 needs `positives.csv` from 01, and 03 needs `dataset_all.csv` from 02. Each script is deterministic (see [Reproducibility](#14-reproducibility)).

**Expected key output**

`01_data_audit.py`

```
Genome length: 4641652 (expected 4,641,652)
confidenceLevel (sigma70): {'W': 1285, 'S': 653, 'C': 55, 'None': 4}
All promoter records 4057 | sigma70 records 2001 | usable TSS and strand 1997
Evidence filter (S, C) 708 | Unique loci 708 | Positive windows (clean) 707
Dropped for sequence mismatch: 1 [('fimAp', '+',