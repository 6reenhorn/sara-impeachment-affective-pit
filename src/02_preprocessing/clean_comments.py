import re
import unicodedata
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/raw/youtube_comments_raw.csv")
OUTPUT_FILE = Path("data/processed/youtube_comments_clean.csv")

MIN_WORDS = 3              # Drops one/two-word noise ("First!", "Amen", "Solid")
MAX_COMMENTS_PER_USER = 5  # Prevents one spammer/troll account from skewing counts

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+")
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
# Keeps letters (including ñ/Ñ and accented characters) and spaces for TF-IDF
NON_WORD_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


def clean_preserve_affect(text: str) -> str:
    """Light cleaning: keeps emojis, capitalization, and punctuation."""
    text = unicodedata.normalize("NFKC", str(text))
    text = URL_PATTERN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_tfidf(text: str) -> str:
    """Normalized text for bag-of-words / TF-IDF and classical ML."""
    text = clean_preserve_affect(text).lower()
    text = MENTION_PATTERN.sub(" ", text)
    text = TIMESTAMP_PATTERN.sub(" ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    text = re.sub(r"\d+", " ", text)
    text = REPEAT_CHAR_PATTERN.sub(r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run Stage 1 first.")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    funnel = [("0. Raw input", len(df))]

    # 1. Drop empty text rows
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].map(clean_preserve_affect)
    df["text_tfidf"] = df["text"].map(clean_for_tfidf)
    df = df[df["text_tfidf"] != ""].copy()
    funnel.append(("1. After removing empty/URL-only", len(df)))

    # 2. Drop exact duplicate comment texts (catches copy-paste spam/bot scripts)
    df = df.drop_duplicates(subset=["text_tfidf"], keep="first").copy()
    funnel.append(("2. After exact-text deduplication", len(df)))

    # 3. Filter ultra-short comments (< MIN_WORDS words)
    df["word_count"] = df["text_tfidf"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS].copy()
    funnel.append((f"3. After keeping >= {MIN_WORDS} words", len(df)))

    # 4. Cap comments per unique user_hash (keep highest-liked ones first)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(by=["likes", "timestamp_utc"], ascending=[False, True])

    known_users = df[df["user_hash"] != "unknown"].groupby(
        "user_hash", group_keys=False
    ).head(MAX_COMMENTS_PER_USER)
    unknown_users = df[df["user_hash"] == "unknown"]
    df = pd.concat([known_users, unknown_users], ignore_index=True)
    funnel.append((f"4. After capping <= {MAX_COMMENTS_PER_USER} comments/user", len(df)))

    # 5. Add simple affect/metadata features useful for EDA later
    df["char_count"] = df["text_clean"].str.len()
    df["exclamation_count"] = df["text_clean"].str.count("!")
    df["question_count"] = df["text_clean"].str.count(r"\?")
    df["upper_ratio"] = df["text_clean"].map(
        lambda s: round(sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1), 3)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("--- Cleaning Funnel ---")
    for step, count in funnel:
        print(f"{step:<45} {count:>5}")

    print("\n--- Clean Dataset by Channel ---")
    print(df["source_name"].value_counts().to_string())
    print(f"\nMedian words/comment: {df['word_count'].median():.1f}")
    print(f"Unique users (hash):  {df['user_hash'].nunique()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()