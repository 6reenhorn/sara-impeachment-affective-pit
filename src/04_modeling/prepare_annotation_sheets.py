"""Stage 3b: Prepare 5-member Human Gold Standard annotation sheets.

Draws 1,050 unique stratified comments:
  - 50 shared calibration comments (included in all 5 sheets for Fleiss' kappa)
  - 1,000 unique comments split across member_1.csv .. member_5.csv (200 each)
Total per member: 250 rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd

LEXICON_FILE = Path("deliverables/lexicon_results.csv")
CLEAN_FILE = Path("data/processed/all_comments_clean.csv")
SHEETS_DIR = Path("data/annotated/sheets")
INDEX_OUT = Path("data/annotated/sample_index.csv")
GUIDELINES_OUT = Path("data/annotated/ANNOTATION_GUIDELINES.md")

RANDOM_STATE = 42
N_MEMBERS = 5
N_SHARED = 50
N_UNIQUE_PER_MEMBER = 200
TOTAL_SAMPLE = N_SHARED + N_MEMBERS * N_UNIQUE_PER_MEMBER  # 1,050


def stratified_sample(df: pd.DataFrame, n: int, strata_cols: list[str], seed: int) -> pd.DataFrame:
    """Proportional stratified sample without replacement, topping up if needed."""
    rng = np.random.default_rng(seed)
    if len(df) <= n:
        return df.copy()

    df = df.copy()
    df["_stratum"] = df[strata_cols].astype(str).agg("|".join, axis=1)
    counts = df["_stratum"].value_counts()
    quotas = (counts / len(df) * n).round().astype(int)

    # Adjust quotas so sum == n
    diff = n - int(quotas.sum())
    order = counts.index.tolist()
    idx = 0
    while diff != 0 and order:
        key = order[idx % len(order)]
        if diff > 0 and quotas[key] < counts[key]:
            quotas[key] += 1
            diff -= 1
        elif diff < 0 and quotas[key] > 1:
            quotas[key] -= 1
            diff += 1
        idx += 1
        if idx > 10000:
            break

    parts = []
    for key, q in quotas.items():
        grp = df[df["_stratum"] == key]
        take = min(int(q), len(grp))
        if take > 0:
            parts.append(grp.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    sampled = pd.concat(parts, ignore_index=False) if parts else df.iloc[0:0]

    if len(sampled) < n:
        remaining = df.drop(index=sampled.index)
        need = min(n - len(sampled), len(remaining))
        sampled = pd.concat([sampled, remaining.sample(n=need, random_state=seed)], ignore_index=False)
    elif len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    return sampled.drop(columns=["_stratum"])


def write_guidelines() -> None:
    content = """# CS412 PIT — 5-Member Human Gold Standard Annotation Guidelines

Thank you for annotating! Each member has **250 comments** (`member_1.csv` .. `member_5.csv`):
- **50 shared calibration comments** (`is_shared_calibration = True`) appear in all 5 files so we can compute **Inter-Annotator Agreement (Fleiss' κ / Cohen's κ)**.
- **200 unique comments** belong only to your file (giving us **1,050 unique gold-labeled comments** total!).

## How to Fill Your CSV
Open your assigned CSV in Excel / Google Sheets (UTF-8) and fill the **`human_label`** column with exactly one of:
- `Positive`
- `Negative`
- `Neutral`

Optional: use the `notes` column to flag **sarcasm**, **Taglish/Bisaya idioms**, or **mixed feelings** (these make great examples for Task 5 & Presentation slides!).

---

## Labeling Rules (Sentiment Valence, NOT Political Side)

> **Golden Rule:** Label the **emotional tone / affective valence expressed by the commenter**, regardless of whether they are pro-Sara or anti-Sara!

1. **`Positive`** — Praise, admiration, gratitude, hope, relief, encouragement, or warm support toward *anyone* (prosecutors, defense lawyers, senators, VP Sara, or the country).
   - *Examples:* `"Salute to Atty. Diokno, very clear and respectful!"`, `"Praying for truth and justice for the Philippines 🙏"`, `"Good job defense team!"`
2. **`Negative`** — Anger, disgust, frustration, mockery/sarcasm, insults, accusations of corruption/lying, fear, or strong disappointment.
   - *Examples:* `"Mga magnanakaw at sinungaling!"`, `"Walang kwenta talaga si Chiz nakakainis"`, `"Wasting taxpayers' money on this circus"`, `"Wow, 'wala po akong naalala' na naman 🙄"` (sarcastic criticism = `Negative`).
3. **`Neutral`** — Factual observations, timestamps, neutral questions, procedural notes, or ambiguous banter without clear positive/negative emotion.
   - *Examples:* `"What time does Day 45 resume?"`, `"Article II covers unexplained wealth."`, `"Nasa hearing si Atty. Cabrera ngayon."`

### Edge Cases
- **Sarcasm / Irony:** Label the **intended** emotion (e.g., `"Galing magpalusot haha 👏"` mocking someone = `Negative`).
- **Mixed Positive + Negative:** Pick the **dominant** emotion in the comment; if truly 50/50 balanced, choose `Neutral` and note `mixed` in `notes`.
"""
    GUIDELINES_OUT.parent.mkdir(parents=True, exist_ok=True)
    GUIDELINES_OUT.write_text(content, encoding="utf-8")


def main():
    if not LEXICON_FILE.exists() or not CLEAN_FILE.exists():
        raise FileNotFoundError("Run Stages 2 and 3 before preparing annotation sheets.")

    lex = pd.read_csv(LEXICON_FILE, encoding="utf-8-sig")
    clean = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")

    # Align on index (both come directly from all_comments_clean.csv in identical order)
    if len(lex) != len(clean):
        raise RuntimeError(f"Row mismatch between {CLEAN_FILE} ({len(clean)}) and {LEXICON_FILE} ({len(lex)})")

    df = clean.copy()
    df["vader_label"] = lex["vader_label"].values
    df["lexicon_majority_label"] = lex["lexicon_majority_label"].values
    df["sample_id"] = [f"SMP-{i+1:05d}" for i in range(len(df))]

    # Balance Reddit representation (~22% of sample if available, capped cleanly)
    n_reddit_target = min(int(round(TOTAL_SAMPLE * 0.22)), int((df["platform"] == "reddit").sum()))
    n_youtube_target = TOTAL_SAMPLE - n_reddit_target

    rd_pool = df[df["platform"] == "reddit"]
    yt_pool = df[df["platform"] != "reddit"]

    rd_samp = stratified_sample(rd_pool, n_reddit_target, ["language_bucket", "vader_label"], RANDOM_STATE)
    yt_samp = stratified_sample(yt_pool, n_youtube_target, ["language_bucket", "vader_label"], RANDOM_STATE + 1)

    total_samp = pd.concat([yt_samp, rd_samp], ignore_index=True)
    total_samp = total_samp.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # Pick 50 shared calibration rows stratified across platform x language_bucket x vader_label
    shared_df = stratified_sample(
        total_samp,
        N_SHARED,
        ["platform", "language_bucket", "vader_label"],
        RANDOM_STATE + 2,
    ).copy()
    shared_ids = set(shared_df["sample_id"])
    unique_df = total_samp[~total_samp["sample_id"].isin(shared_ids)].copy()
    unique_df = unique_df.sample(frac=1.0, random_state=RANDOM_STATE + 3).reset_index(drop=True)

    if len(unique_df) != N_MEMBERS * N_UNIQUE_PER_MEMBER:
        raise RuntimeError(f"Expected {N_MEMBERS * N_UNIQUE_PER_MEMBER} unique rows, got {len(unique_df)}")

    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    sheet_cols = ["sample_id", "is_shared_calibration", "platform", "source_name", "text_clean", "human_label", "notes"]

    for m in range(N_MEMBERS):
        start = m * N_UNIQUE_PER_MEMBER
        end = start + N_UNIQUE_PER_MEMBER
        u_slice = unique_df.iloc[start:end].copy()
        u_slice["is_shared_calibration"] = False

        s_slice = shared_df.copy()
        s_slice["is_shared_calibration"] = True

        member_df = pd.concat([s_slice, u_slice], ignore_index=True)
        member_df["human_label"] = ""
        member_df["notes"] = ""
        member_df = member_df[sheet_cols].sample(frac=1.0, random_state=RANDOM_STATE + 10 + m).reset_index(drop=True)

        out_path = SHEETS_DIR / f"member_{m + 1}.csv"
        member_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Save index mapping back to full features
    total_samp["is_shared_calibration"] = total_samp["sample_id"].isin(shared_ids)
    total_samp["assigned_member"] = "shared_all_5"
    for m in range(N_MEMBERS):
        start = m * N_UNIQUE_PER_MEMBER
        end = start + N_UNIQUE_PER_MEMBER
        ids_m = set(unique_df.iloc[start:end]["sample_id"])
        total_samp.loc[total_samp["sample_id"].isin(ids_m), "assigned_member"] = f"member_{m + 1}"

    index_cols = [
        "sample_id",
        "assigned_member",
        "is_shared_calibration",
        "platform",
        "source_name",
        "comment_id",
        "language_bucket",
        "text_clean",
        "text_lemmatized",
        "vader_label",
    ]
    total_samp[index_cols].to_csv(INDEX_OUT, index=False, encoding="utf-8-sig")
    write_guidelines()

    print("=== Stage 3b: 5-Member Annotation Sheets Prepared ===")
    print(f"Total unique sampled comments: {len(total_samp):,} ({N_SHARED} shared + {N_MEMBERS} x {N_UNIQUE_PER_MEMBER} unique)")
    print("Platform split:")
    print(total_samp["platform"].value_counts().to_string())
    print("\nLanguage split:")
    print(total_samp["language_bucket"].value_counts().to_string())
    print(f"\nSheets written to: {SHEETS_DIR}/member_1.csv .. member_5.csv (250 rows each)")
    print(f"Guidelines saved:  {GUIDELINES_OUT}")


if __name__ == "__main__":
    main()
