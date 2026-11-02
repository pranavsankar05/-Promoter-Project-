import random
import re
from pathlib import Path
import numpy as np
import pandas as pd

RAW, OUT, RES = Path("data/raw"), Path("data/processed"), Path("results")
GENOME_FILE = RAW / "NC_000913.3.fasta"
PROMOTER_FILE = RAW / "PromoterSet.tsv"
POS_FILE = OUT / "positives.csv"

SEED = 42
N_GC = 2                  # background windows per promoter
GC_TOL = 0.02             # allowed GC difference from the promoter
ZONE = 100                # stay this many bp away from every known TSS
MAX_TRIES = 20000
BASES = set("ACGT")
COMP = str.maketrans("ACGT", "TGCA")

def revcomp(s): return s.translate(COMP)[::-1]
def gc(s): return (s.count("G") + s.count("C")) / len(s)

def read_genome():
    lines = GENOME_FILE.read_text().splitlines()
    return "".join(l.strip() for l in lines if not l.startswith(">")).upper()

def all_tss_positions():
    lines = PROMOTER_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    h = next(i for i, l in enumerate(lines)
             if "postss" in l.lower() and "sigmafactor" in l.lower())
    cols = [re.sub(r"^[#\s]*(\d+\)\s*)?", "", c).strip() for c in lines[h].split("\t")]
    idx = cols.index("posTSS")
    out = []
    for l in lines[h + 1:]:
        if not l.strip() or l.startswith("#"):
            continue
        r = l.split("\t")
        try: out.append(int(float(r[idx].strip())))
        except (ValueError, IndexError): pass
    return out

def build_prefix(L, tss_list):
    mask = np.zeros(L, dtype=np.uint8)
    for p in tss_list:
        for x in range(p - ZONE - 1, p + ZONE):
            mask[x % L] = 1
    return np.concatenate([[0], np.cumsum(mask, dtype=np.int32)])

def sample_gc_matched(g, prefix, target, W, rng, used):
    L = len(g)
    for _ in range(MAX_TRIES):
        s0 = rng.randrange(0, L - W)
        if prefix[s0 + W] - prefix[s0] or s0 in used:
            continue
        seq = g[s0:s0 + W]
        if not set(seq) <= BASES or abs(gc(seq) - target) > GC_TOL:
            continue
        strand = rng.choice("+-")
        used.add(s0)
        return {"sequence": revcomp(seq) if strand == "-" else seq, "label": 0,
                "strand": strand, "genomic_start": s0 + 1, "genomic_end": s0 + W,
                "bg_type": "gc_matched", "source": "selected_background"}
    return None

def main():
    rng = random.Random(SEED)
    g = read_genome()
    pos = pd.read_csv(POS_FILE, keep_default_na=False)
    W = len(pos["sequence"].iloc[0])
    prefix = build_prefix(len(g), all_tss_positions())
    used, rows, failed = set(), [], 0
    for _, p in pos.iterrows():
        target = gc(p["sequence"])
        for _ in range(N_GC):
            r = sample_gc_matched(g, prefix, target, W, rng, used)
            if r: rows.append(r)
            else: failed += 1
    neg = pd.DataFrame(rows)
    neg.insert(0, "sequence_id", [f"neg_{i:05d}" for i in range(1, len(neg) + 1)])
    pos = pos.assign(bg_type="promoter")
    data = pd.concat([pos, neg], ignore_index=True).fillna("")
    data.to_csv(OUT / "dataset_all.csv", index=False)

    print(f"Window length: {W} bp | positives: {len(pos)} | negatives: {len(neg)}")
    print(f"Failed samples: {failed}")
    print(f"Positive share: {len(pos) / len(data):.2%}")
    print("\nMean GC:")
    
    print(f"  promoter    {pos['sequence'].map(gc).mean():.3f}")
    print(f"  gc_matched  {neg['sequence'].map(gc).mean():.3f}")
    lo, hi = 60 - 14, 60 - 5
    print("\nMean AT fraction at positions -14..-6:")
    for name, df in (("promoter", pos), ("gc_matched", neg)):
        at = df["sequence"].map(lambda s: sum(c in "AT" for c in s[lo:hi]) / (hi - lo)).mean()
        print(f"  {name:<11}{at:.3f}")

if __name__ == "__main__":
    main()