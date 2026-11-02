import re
import urllib.request
from pathlib import Path
import pandas as pd

RAW, OUT, RES = Path("data/raw"), Path("data/processed"), Path("results")
for p in (RAW, OUT, RES):
    p.mkdir(parents=True, exist_ok=True)

PROMOTER_FILE = RAW / "PromoterSet.tsv"
GENOME_FILE = RAW / "NC_000913.3.fasta"
UP, DOWN = 60, 20                # window = UP + 1 (TSS) + DOWN = 81 bp, TSS at index UP
KEEP_CONFIDENCE = ("S", "C")     # RegulonDB confidenceLevel: S, C kept; W (weak) excluded

COMP = str.maketrans("ACGT", "TGCA")
def revcomp(s): return s.translate(COMP)[::-1]

def download_genome():
    if not GENOME_FILE.exists():
        url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
               "?db=nuccore&id=NC_000913.3&rettype=fasta&retmode=text")
        urllib.request.urlretrieve(url, GENOME_FILE)
    lines = GENOME_FILE.read_text().splitlines()
    g = "".join(l.strip() for l in lines if not l.startswith(">")).upper()
    print("Genome length:", len(g), "(expected 4,641,652)")
    return g

def get_slice(g, start, end):          # circular genome
    L = len(g)
    if start < 0: return g[start:] + g[:end]
    if end > L:   return g[start:] + g[:end - L]
    return g[start:end]

def load_promoters():
    lines = PROMOTER_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    # header = first line (even if it starts with '#') that mentions posTSS and sigmaFactor
    h = next((i for i, l in enumerate(lines)
              if "postss" in l.lower() and "sigmafactor" in l.lower()), None)
    if h is None:
        print("Header not found. First 5 lines of the file:")
        for l in lines[:5]: print(repr(l[:200]))
        raise SystemExit("Fix: re-download PromoterSet from RegulonDB, or paste these lines to me.")
    cols = [re.sub(r"^[#\s]*(\d+\)\s*)?", "", c).strip() for c in lines[h].split("\t")]
    rows = []
    for l in lines[h + 1:]:
        if not l.strip() or l.startswith("#"):
            continue
        r = l.split("\t")
        rows.append((r + [""] * len(cols))[:len(cols)])
    df = pd.DataFrame(rows, columns=cols).apply(lambda c: c.str.strip())
    # your file uses pm* names; rename them to the names the rest of the script uses
    df = df.rename(columns={"pmId": "id", "pmName": "name", "pmSequence": "sequence",
                            "pmEvidence": "promoterEvidence"})
    df["posTSS"] = pd.to_numeric(df["posTSS"], errors="coerce")
    df["strand"] = df["strand"].str.lower().map({"forward": "+", "reverse": "-"})
    print("Columns:", list(df.columns))
    return df

def main():
    g = download_genome()
    df = load_promoters()
    counts = [("All promoter records", len(df))]
    s70 = df[df["sigmaFactor"].str.lower().str.contains("sigma70", na=False)]
    counts.append(("sigma70 records", len(s70)))
    ok = s70[s70["posTSS"].notna() & s70["strand"].notna()]
    counts.append(("With usable TSS and strand", len(ok)))
    print("confidenceLevel (sigma70):", ok["confidenceLevel"].value_counts().to_dict())
    strong = ok[ok["confidenceLevel"].isin(KEEP_CONFIDENCE)]
    counts.append((f"Evidence filter (confidence in {KEEP_CONFIDENCE})", len(strong)))
    uniq = strong.drop_duplicates(subset=["strand", "posTSS"])
    counts.append(("Unique loci (strand, TSS)", len(uniq)))

    recs, mismatches = [], []
    for i, r in enumerate(uniq.itertuples(), 1):
        p = int(r.posTSS)                    # 1-based genomic position of +1
        if r.strand == "+":
            seq = get_slice(g, p - 1 - UP, p + DOWN)
            gs, ge = p - UP, p + DOWN
        else:
            seq = revcomp(get_slice(g, p - DOWN - 1, p + UP))
            gs, ge = p - DOWN, p + UP
        if not (set(seq) <= set("ACGT") and len(seq) == UP + 1 + DOWN):
            continue
        if r.sequence and seq != r.sequence.upper():
            mismatches.append((r.name, r.strand, p))
            continue                         # drop windows that disagree with RegulonDB
        recs.append({"sequence_id": f"pos_{i:05d}", "sequence": seq, "label": 1,
                     "strand": r.strand, "genomic_start": gs, "genomic_end": ge,
                     "sigma_factor": r.sigmaFactor, "evidence_type": r.promoterEvidence,
                     "confidence": r.confidenceLevel, "box10_seq": r.boxMinus10seq,
                     "box35_seq": r.boxMinus35seq, "source": "RegulonDB"})
    pos = pd.DataFrame(recs)
    counts.append(("Positive windows (clean)", len(pos)))
    pos.to_csv(OUT / "positives.csv", index=False)
    pd.DataFrame(counts, columns=["step", "count"]).to_csv(RES / "audit_counts.csv", index=False)
    print("\nAUDIT COUNTS")
    for a, b in counts: print(f"  {a:<45}{b}")

    n = len(pos)
    print(f"\nDropped for sequence mismatch: {len(mismatches)}", mismatches[:5])
    print("With annotated -10 box:", int((pos['box10_seq'] != '').sum()),
          "| with annotated -35 box:", int((pos['box35_seq'] != '').sum()))

    at = [sum(s[i] in "AT" for s in pos["sequence"]) / n for i in range(UP + 1 + DOWN)]
    mean_at = sum(at[:UP]) / UP
    print(f"\nMean AT fraction, upstream window: {mean_at:.2f}")
    for off in range(-14, -4):
        print(f"  position {off}: AT = {at[UP + off]:.2f}")
    peak = max(at[UP - 13:UP - 6])
    print("AT check:", "OK" if peak > mean_at + 0.10 else "WARNING: -10 region not AT-rich")

if __name__ == "__main__":
    main()