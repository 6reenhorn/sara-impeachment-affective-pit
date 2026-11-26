"""Merge completed 5-member annotation sheets into data/annotated/gold_labels.csv.

Validates labels, computes 5-rater Fleiss' kappa and pairwise Cohen's kappa on the
50 shared calibration comments, resolves shared calibration labels via majority vote,
and outputs data/annotated/gold_labels.csv + inter_annotator_report.txt.
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

SHEETS_DIR = Path("data/annotated/sheets")
INDEX_FILE = Path("data/annotated/sample_index.csv")
GOLD_OUT = Path("data/annotated/gold_labels.csv")
REPORT_OUT = Path("data/annotated/inter_annotator_report.txt")

CATEGORIES = ["Negative", "Neutral", "Positive"]
ALIAS_MAP = {
    "pos": "Positive",
    "positive": "Positive",
    "+": "Positive",
    "1": "Positive",
    "neg": "Negative",
    "negative": "Negative",
    "-": "Negative",
    "-1": "Negative",
    "neu": "Neutral",
    "neutral": "Neutral",
    "0": "Neutral",
}


def normalize_label(val) -> str:
    s = str(val or "").strip().lower()
    if s in ALIAS_MAP:
        return ALIAS_MAP[s]
    return ""


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for N subjects x K categories count matrix."""
    n_sub, n_cat = ratings_matrix.shape
    n_raters = int(ratings_matrix.sum(axis=1)[0])
    if n_sub == 0 or n_raters <= 1:
        return float("nan")

    p_j = ratings_matrix.sum(axis=0) / (n_sub * n_raters)
    p_e = float(np.sum(p_j ** 2))
    p_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(np.mean(p_i))
    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def main():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"Missing {INDEX_FILE}. Run prepare_annotation_sheets.py first.")

    index_df = pd.read_csv(INDEX_FILE, encoding="utf-8-sig")
    member_files = [SHEETS_DIR / f"member_{i}.csv" for i in range(1, 6)]
    for mf in member_files:
        if not mf.exists():
            raise FileNotFoundError(f"Missing annotation sheet: {mf}")

    sheets = []
    for i, mf in enumerate(member_files, start=1):
        df = pd.read_csv(mf, encoding="utf-8-sig")
        df["annotator"] = f"member_{i}"
        df["norm_label"] = df["human_label"].map(normalize_label)
        unlabeled = (df["norm_label"] == "").sum()
        if unlabeled > 0:
            raise ValueError(
                f"{mf.name} still has {unlabeled} unlabeled or invalid rows. "
                "Fill all human_label cells with Positive, Negative, or Neutral before merging."
            )
        sheets.append(df)

    all_ann = pd.concat(sheets, ignore_index=True)

    # Inter-annotator agreement on shared calibration rows
    shared = all_ann[all_ann["is_shared_calibration"].astype(bool)].copy()
    piv = shared.pivot(index="sample_id", columns="annotator", values="norm_label")

    cat_to_idx = {c: idx for idx, c in enumerate(CATEGORIES)}
    counts_mat = np.zeros((len(piv), len(CATEGORIES)), dtype=int)
    for r_idx, (_, row) in enumerate(piv.iterrows()):
        for val in row.values:
            counts_mat[r_idx, cat_to_idx[val]] += 1

    fk = fleiss_kappa(counts_mat)

    lines = [
        "=== 5-Member Inter-Annotator Agreement Report ===",
        f"Shared calibration items: {len(piv)}",
        f"5-Rater Fleiss' Kappa:    {fk:.4f}",
        "\nPairwise Cohen's Kappa:",
    ]
    members = [f"member_{i}" for i in range(1, 6)]
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            k = cohen_kappa_score(piv[members[i]], piv[members[j]], labels=CATEGORIES)
            pct = (piv[members[i]] == piv[members[j]]).mean() * 100
            lines.append(f"  {members[i]} vs {members[j]}: kappa = {k:.4f} (raw match = {pct:.1f}%)")

    # Resolve shared items by majority vote (ties resolved via Neutral -> Negative -> Positive order)
    shared_resolved = {}
    ties = 0
    for sid, row in piv.iterrows():
        cnt = Counter(row.values)
        top_cnt = cnt.most_common(1)[0][1]
        tied_labels = [lbl for lbl, c in cnt.items() if c == top_cnt]
        if len(tied_labels) > 1:
            ties += 1
            chosen = "Neutral" if "Neutral" in tied_labels else tied_labels[0]
        else:
            chosen = tied_labels[0]
        shared_resolved[sid] = chosen
    lines.append(f"\nCalibration tie-breaks resolved: {ties}")

    unique_ann = all_ann[~all_ann["is_shared_calibration"].astype(bool)].copy()
    unique_map = dict(zip(unique_ann["sample_id"], unique_ann["norm_label"]))
    combined_map = {**unique_map, **shared_resolved}

    gold_df = index_df.copy()
    gold_df["gold_label"] = gold_df["sample_id"].map(combined_map)
    if gold_df["gold_label"].isna().any():
        missing = int(gold_df["gold_label"].isna().sum())
        raise RuntimeError(f"{missing} sample_ids in sample_index.csv were missing from member sheets.")

    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
    gold_df.to_csv(GOLD_OUT, index=False, encoding="utf-8-sig")

    lines.append(f"\nMerged Gold Standard Rows: {len(gold_df):,}")
    lines.append("Label Distribution:")
    for lbl in CATEGORIES:
        c = int((gold_df["gold_label"] == lbl).sum())
        lines.append(f"  {lbl:<10}: {c:>4} ({c / len(gold_df) * 100:.1f}%)")

    report = "\n".join(lines) + "\n"
    REPORT_OUT.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved gold labels to: {GOLD_OUT}")


if __name__ == "__main__":
    main()
