from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

OUT, RES = Path("data/processed"), Path("results")
RES.mkdir(exist_ok=True)

BOX = 6
FIND10 = (38, 60)
FIND35 = (15, 45)
BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}
KMERS = ["".join(k) for k in product(BASES, repeat=3)]
KIDX = {k: i for i, k in enumerate(KMERS)}
N_FOLDS = 5          # contiguous blocks along the genome
BUFFER = 200         # bp removed from training around each test block
N_BOOT = 1000
SEED = 42
MODELS = ["Baseline", "A", "B", "C"]
PAIRS = [("B", "A"), ("B", "C"), ("B", "Baseline"), ("C", "A")]

# ---------- motif (PWM), same method as 03_model.py ----------
def box_start(seq, box, lo, hi):
    box = box.upper()
    if len(box) != BOX or set(box) - set(BASES):
        return None
    i = seq.find(box, lo, hi)
    return i if i >= 0 else None

def collect(pos_df, col, lo, hi):
    kmers, starts = [], []
    for s, b in zip(pos_df["sequence"], pos_df[col]):
        i = box_start(s, b, lo, hi)
        if i is not None:
            kmers.append(s[i:i + BOX])
            starts.append(i)
    return kmers, starts

def bg_freq(seqs):
    allseq = "".join(seqs)
    c = np.array([allseq.count(b) for b in BASES], float)
    return c / c.sum()

def make_pwm(kmers, bg):
    counts = np.ones((BOX, 4))
    for k in kmers:
        for j, c in enumerate(k):
            counts[j, IDX[c]] += 1
    freq = counts / counts.sum(axis=1, keepdims=True)
    return np.log2(freq / bg)

def scan(seq, pwm, a, b):
    best, pos = -1e9, a
    for s in range(a, b + 1):
        sc = sum(pwm[j, IDX[c]] for j, c in enumerate(seq[s:s + BOX]))
        if sc > best:
            best, pos = sc, s
    return best, pos

def motif_feats(seqs, pwm10, r10, pwm35, r35):
    rows = []
    for s in seqs:
        s10, p10 = scan(s, pwm10, *r10)
        s35, p35 = scan(s, pwm35, *r35)
        rows.append([s10, s35, p10 - (p35 + BOX)])
    return np.array(rows)

# ---------- other features ----------
def comp_feats(seqs):
    rows = []
    for s in seqs:
        n = len(s)
        nuc = [s.count(b) / n for b in BASES]
        gc = (s.count("G") + s.count("C")) / n
        k = np.zeros(64)
        for i in range(n - 2):
            k[KIDX[s[i:i + 3]]] += 1
        rows.append(nuc + [gc] + list(k / (n - 2)))
    return np.array(rows)

def onehot_feats(seqs):
    n = len(seqs[0])
    arr = np.zeros((len(seqs), n * 4))
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            arr[i, j * 4 + IDX[c]] = 1
    return arr

def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]

def main():
    df = pd.read_csv(OUT / "dataset_all.csv", keep_default_na=False)
    df = df[df["bg_type"].isin(["promoter", "gc_matched"])].copy()
    df["genomic_start"] = pd.to_numeric(df["genomic_start"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("genomic_start").reset_index(drop=True)
    seqs, y = df["sequence"].tolist(), df["label"].values
    starts = df["genomic_start"].values
    n = len(df)
    print(f"Rows: {n} | promoters: {int(y.sum())} ({y.mean():.1%}) | folds: {N_FOLDS}")

    C_all, O_all = comp_feats(seqs), onehot_feats(seqs)
    fold_id = np.minimum(np.arange(n) * N_FOLDS // n, N_FOLDS - 1)
    oof = {m: np.full(n, np.nan) for m in MODELS}
    rows = []

    for k in range(N_FOLDS):
        te = np.where(fold_id == k)[0]
        lo, hi = starts[te].min(), starts[te].max()
        tr = np.where((fold_id != k) & ((starts < lo - BUFFER) | (starts > hi + BUFFER)))[0]
        ytr, yte = y[tr], y[te]
        tr_pos = df.iloc[tr][df.iloc[tr]["label"] == 1]
        bg = bg_freq([seqs[i] for i in tr if y[i] == 0])
        k10, st10 = collect(tr_pos, "box10_seq", *FIND10)
        k35, st35 = collect(tr_pos, "box35_seq", *FIND35)
        if len(k10) < 50 or len(k35) < 50:
            raise SystemExit(f"Fold {k}: too few boxes ({len(k10)}, {len(k35)}). Paste this to me.")
        pwm10, pwm35 = make_pwm(k10, bg), make_pwm(k35, bg)
        r10 = (int(np.percentile(st10, 5)), int(np.percentile(st10, 95)))
        r35 = (int(np.percentile(st35, 5)), int(np.percentile(st35, 95)))
        M_tr = motif_feats([seqs[i] for i in tr], pwm10, r10, pwm35, r35)
        M_te = motif_feats([seqs[i] for i in te], pwm10, r10, pwm35, r35)

        scores = {"Baseline": M_te[:, 0] + M_te[:, 1],
                  "A": fit_predict(C_all[tr], ytr, C_all[te]),
                  "B": fit_predict(np.hstack([C_all[tr], M_tr]), ytr, np.hstack([C_all[te], M_te])),
                  "C": fit_predict(np.hstack([C_all[tr], O_all[tr]]), ytr, np.hstack([C_all[te], O_all[te]]))}
        row = {"fold": k + 1, "n_train": len(tr), "n_test": len(te),
               "test_promoter_share": round(float(yte.mean()), 3),
               "boxes_-10/-35": f"{len(k10)}/{len(k35)}"}
        for m in MODELS:
            oof[m][te] = scores[m]
            row[m] = round(average_precision_score(yte, scores[m]), 4)
        rows.append(row)
        print(f"Fold {k + 1} done: " + ", ".join(f"{m}={row[m]:.3f}" for m in MODELS))

    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(RES / "cv_fold_results.csv", index=False)
    print("\nPER-FOLD PR-AUC")
    print(fold_table.to_string(index=False))

    # mean and spread across folds
    summ = pd.DataFrame({"mean_PR_AUC": [fold_table[m].mean() for m in MODELS],
                         "sd_across_folds": [fold_table[m].std(ddof=1) for m in MODELS],
                         "pooled_PR_AUC": [average_precision_score(y, oof[m]) for m in MODELS]},
                        index=MODELS).round(4)
    summ.to_csv(RES / "cv_summary.csv")
    print(f"\nSUMMARY (random-guess level = overall promoter share {y.mean():.3f})")
    print(summ.to_string())

    # paired differences: per fold, and bootstrap on pooled out-of-fold predictions
    rng = np.random.default_rng(SEED)
    boot = {m: [] for m in MODELS}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        for m in MODELS:
            boot[m].append(average_precision_score(y[idx], oof[m][idx]))
    boot = {m: np.array(v) for m, v in boot.items()}
    prow = []
    for a, b in PAIRS:
        d_fold = fold_table[a] - fold_table[b]
        d_boot = boot[a] - boot[b]
        prow.append({"comparison": f"{a} minus {b}",
                     "mean_fold_diff": round(d_fold.mean(), 4),
                     "folds_a_ahead": f"{int((d_fold > 0).sum())}/{N_FOLDS}",
                     "pooled_diff": round(average_precision_score(y, oof[a]) - average_precision_score(y, oof[b]), 4),
                     "pooled_CI_low": round(np.percentile(d_boot, 2.5), 4),
                     "pooled_CI_high": round(np.percentile(d_boot, 97.5), 4)})
    ptable = pd.DataFrame(prow)
    ptable.to_csv(RES / "cv_paired_diffs.csv", index=False)
    print("\nPAIRED DIFFERENCES")
    print(ptable.to_string(index=False))
    print(f"\nBootstrap: {len(boot['A'])} resamples of the pooled out-of-fold predictions, seed {SEED}.")
    print("Saved cv_fold_results.csv, cv_summary.csv and cv_paired_diffs.csv in results/")

if __name__ == "__main__":
    main()