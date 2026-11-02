from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)

OUT, RES = Path("data/processed"), Path("results")
RES.mkdir(exist_ok=True)

BOX = 6                       # length of each motif box
FIND10 = (38, 60)             # where to look for the annotated -10 box in a window
FIND35 = (15, 45)             # where to look for the annotated -35 box
BASES = "ACGT"
IDX = {b: i for i, b in enumerate(BASES)}
KMERS = ["".join(k) for k in product(BASES, repeat=3)]
KIDX = {k: i for i, k in enumerate(KMERS)}
N_BOOT = 1000                 # bootstrap resamples of the test set
SEED = 42

# ---------- data ----------
def load_split():
    df = pd.read_csv(OUT / "dataset_all.csv", keep_default_na=False)
    df = df[df["bg_type"].isin(["promoter", "gc_matched"])].copy()
    df["genomic_start"] = pd.to_numeric(df["genomic_start"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values("genomic_start").reset_index(drop=True)
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

# ---------- motif (PWM) ----------
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
    counts = np.ones((BOX, 4))                    # pseudocount of 1
    for k in kmers:
        for j, c in enumerate(k):
            counts[j, IDX[c]] += 1
    freq = counts / counts.sum(axis=1, keepdims=True)
    return np.log2(freq / bg)                     # log-odds vs background

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
        rows.append([s10, s35, p10 - (p35 + BOX)])   # spacing between the two boxes
    return np.array(rows)

# ---------- composition ----------
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

# ---------- position-aware control (model C) ----------
def onehot_feats(seqs):
    n = len(seqs[0])
    arr = np.zeros((len(seqs), n * 4))
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            arr[i, j * 4 + IDX[c]] = 1
    return arr

# ---------- evaluation ----------
def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return t[np.argmax(f)]

def metrics(name, y, score, thr):
    pred = (score >= thr).astype(int)
    return {"model": name,
            "PR_AUC": round(average_precision_score(y, score), 4),
            "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred), 4),
            "F1": round(f1_score(y, pred), 4),
            "threshold": round(float(thr), 4)}

def bootstrap(y, curves, n_boot, seed):
    """Resample test rows with replacement; same rows for every model (paired)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = {name: [] for name in curves}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        for name, s in curves.items():
            boot[name].append(average_precision_score(y[idx], s[idx]))
    return {k: np.array(v) for k, v in boot.items()}

def main():
    train, test = load_split()
    ytr, yte = train["label"].values, test["label"].values
    print(f"Train: {len(train)} rows ({ytr.mean():.1%} promoters) | "
          f"Test: {len(test)} rows ({yte.mean():.1%} promoters)")

    # fit PWMs on TRAINING promoters only
    tr_pos = train[train["label"] == 1]
    bg = bg_freq(train.loc[train["label"] == 0, "sequence"])
    k10, st10 = collect(tr_pos, "box10_seq", *FIND10)
    k35, st35 = collect(tr_pos, "box35_seq", *FIND35)
    print(f"Training boxes used: -10 = {len(k10)}, -35 = {len(k35)}")
    if len(k10) < 50 or len(k35) < 50:
        raise SystemExit("Too few boxes found. Paste this output to me.")
    pwm10, pwm35 = make_pwm(k10, bg), make_pwm(k35, bg)
    r10 = (int(np.percentile(st10, 5)), int(np.percentile(st10, 95)))
    r35 = (int(np.percentile(st35, 5)), int(np.percentile(st35, 95)))
    print(f"Scan regions (window index): -10 = {r10}, -35 = {r35}")

    seq_tr, seq_te = train["sequence"].tolist(), test["sequence"].tolist()
    M_tr = motif_feats(seq_tr, pwm10, r10, pwm35, r35)
    M_te = motif_feats(seq_te, pwm10, r10, pwm35, r35)
    C_tr, C_te = comp_feats(seq_tr), comp_feats(seq_te)
    O_tr, O_te = onehot_feats(seq_tr), onehot_feats(seq_te)

    results, curves = [], {}

    # baseline: -10 score + -35 score (threshold picked on the training set)
    base = "Baseline (PWM sum)"
    b_tr, b_te = M_tr[:, 0] + M_tr[:, 1], M_te[:, 0] + M_te[:, 1]
    results.append(metrics(base, yte, b_te, best_f1_threshold(ytr, b_tr)))
    curves[base] = b_te

    # A: composition | B: composition + motif | C: composition + position one-hot
    models = (("A: composition", C_tr, C_te),
              ("B: composition + motif", np.hstack([C_tr, M_tr]), np.hstack([C_te, M_te])),
              ("C: composition + position one-hot", np.hstack([C_tr, O_tr]), np.hstack([C_te, O_te])))
    for name, Xtr, Xte in models:
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(sc.transform(Xtr), ytr)
        prob = lr.predict_proba(sc.transform(Xte))[:, 1]
        results.append(metrics(name, yte, prob, 0.5))
        curves[name] = prob

    # bootstrap intervals on the test set
    boot = bootstrap(yte, curves, N_BOOT, SEED)
    n_ok = len(next(iter(boot.values())))
    table = pd.DataFrame(results)
    table["PR_AUC_95CI_low"] = [round(np.percentile(boot[m], 2.5), 4) for m in table["model"]]
    table["PR_AUC_95CI_high"] = [round(np.percentile(boot[m], 97.5), 4) for m in table["model"]]
    table.to_csv(RES / "results_table.csv", index=False)

    print("\nRESULTS (held-out test set)")
    print(table.to_string(index=False))
    print(f"\nRandom-guess PR-AUC = test promoter share = {yte.mean():.3f}")
    print("Thresholds: A, B, C use 0.5; the baseline uses the F1-best value on train.")
    print(f"Bootstrap: {n_ok} resamples of the test set, seed {SEED}, percentile 95% intervals.")

    # paired differences (same resamples for both models)
    pairs = [("B: composition + motif", "A: composition"),
             ("B: composition + motif", "C: composition + position one-hot"),
             ("B: composition + motif", base),
             ("C: composition + position one-hot", "A: composition")]
    prow = []
    for x, y in pairs:
        d = boot[x] - boot[y]
        prow.append({"comparison": f"{x.split(':')[0]} minus {y.split(':')[0]}",
                     "PR_AUC_diff": round(average_precision_score(yte, curves[x])
                                          - average_precision_score(yte, curves[y]), 4),
                     "CI_low": round(np.percentile(d, 2.5), 4),
                     "CI_high": round(np.percentile(d, 97.5), 4),
                     "share_of_resamples_gap_positive": round(float((d > 0).mean()), 3)})
    ptable = pd.DataFrame(prow)
    ptable.to_csv(RES / "results_paired_diffs.csv", index=False)
    print("\nPAIRED PR-AUC DIFFERENCES (95% bootstrap interval)")
    print(ptable.to_string(index=False))

    plt.figure(figsize=(6.5, 5))
    for name, s in curves.items():
        p, r, _ = precision_recall_curve(yte, s)
        plt.plot(r, p, label=f"{name.split(':')[0] if ':' in name else 'Baseline'} "
                             f"(AP={average_precision_score(yte, s):.3f})")
    plt.axhline(yte.mean(), ls="--", color="gray", label="Random guess")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall on held-out test set")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(RES / "pr_curve.png", dpi=200)
    print("\nSaved results_table.csv, results_paired_diffs.csv and pr_curve.png in results/")

if __name__ == "__main__":
    main()