import re
from pathlib import Path
import pandas as pd

RAW, RES = Path("data/raw"), Path("results")
RES.mkdir(exist_ok=True)
PROMOTER_FILE = RAW / "PromoterSet.tsv"

def load_promoters():
    lines = PROMOTER_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    h = next(i for i, l in enumerate(lines)
             if "postss" in l.lower() and "sigmafactor" in l.lower())
    cols = [re.sub(r"^[#\s]*(\d+\)\s*)?", "", c).strip() for c in lines[h].split("\t")]
    rows = []
    for l in lines[h + 1:]:
        if not l.strip() or l.startswith("#"):
            continue
        r = l.split("\t")
        rows.append((r + [""] * len(cols))[:len(cols)])
    df = pd.DataFrame(rows, columns=cols).apply(lambda c: c.str.strip())
    df["posTSS"] = pd.to_numeric(df["posTSS"], errors="coerce")
    df["strand"] = df["strand"].str.lower().map({"forward": "+", "reverse": "-"})
    return df

def main():
    df = load_promoters()
    df["sigma"] = df["sigmaFactor"].str.lower().replace("", "none")
    df["confidenceLevel"] = df["confidenceLevel"].replace("", "none")
    ok = df[df["posTSS"].notna() & df["strand"].notna()].copy()
    print(f"All records: {len(df)} | with usable TSS and strand: {len(ok)}")

    # 1) records per sigma label and confidence level
    table = pd.crosstab(ok["sigma"], ok["confidenceLevel"], margins=True)
    print("\nRECORDS BY SIGMA LABEL AND CONFIDENCE")
    print(table.sort_values("All", ascending=False).to_string())

    # 2) unique loci (strand, TSS) per sigma label, all confidence and strong only
    uniq = ok.drop_duplicates(subset=["sigma", "strand", "posTSS"])
    strong = uniq[uniq["confidenceLevel"].isin(["S", "C"])]
    summary = pd.DataFrame({"unique_loci_all": uniq["sigma"].value_counts(),
                            "unique_loci_S_or_C": strong["sigma"].value_counts()}).fillna(0).astype(int)
    summary = summary.sort_values("unique_loci_all", ascending=False)
    print("\nUNIQUE LOCI PER SIGMA LABEL")
    print(summary.to_string())

    # 3) loci that carry more than one sigma label
    multi = ok.groupby(["strand", "posTSS"])["sigma"].nunique()
    print(f"\nLoci with more than one sigma label: {int((multi > 1).sum())} "
          f"of {len(multi)} unique loci")

    # 4) annotated -10 and -35 boxes among strong loci
    has10 = (strong["boxMinus10seq"] != "").sum()
    has35 = (strong["boxMinus35seq"] != "").sum()
    print(f"Strong loci with annotated -10 box: {has10} | with -35 box: {has35}")

    table.to_csv(RES / "sigma_counts_records.csv")
    summary.to_csv(RES / "sigma_counts_loci.csv")
    print("\nSaved results/sigma_counts_records.csv and results/sigma_counts_loci.csv")

if __name__ == "__main__":
    main()