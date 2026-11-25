"""Harmonize multi-platform raw datasets and export PII-scrubbed deliverables/raw_comments.csv."""

import html
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

YOUTUBE_RAW = Path("data/raw/youtube_comments_raw.csv")
REDDIT_RAW = Path("data/raw/reddit_comments_raw.csv")
EXTERNAL_DIR = Path("data/external")

ALL_RAW_OUT = Path("data/raw/all_comments_raw.csv")
DELIVERABLE_RAW_OUT = Path("deliverables/raw_comments.csv")

CANONICAL_COLUMNS = [
    "platform",
    "source_name",
    "topic_tag",
    "source_id",
    "comment_id",
    "user_hash",
    "timestamp_utc",
    "likes",
    "text",
]

DELIVERABLE_COLUMNS = ["platform", "timestamp_utc", "likes", "text"]

AT_MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-]+")
REDDIT_USER_RE = re.compile(r"(?:^|(?<=\s))/?u/[A-Za-z0-9_\-]+", re.IGNORECASE)


def scrub_pii_text(text: str) -> str:
    """Unescape HTML entities and strip inline @username and u/username mentions."""
    cleaned = html.unescape(str(text or ""))
    cleaned = AT_MENTION_RE.sub(" ", cleaned)
    cleaned = REDDIT_USER_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def format_epoch_utc(series: pd.Series) -> pd.Series:
    """Convert Unix epoch seconds or timestamp strings to standardized UTC strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series("", index=series.index, dtype="object")
    is_num = numeric.notna()
    if is_num.any():
        dt = pd.to_datetime(numeric[is_num], unit="s", utc=True, errors="coerce")
        out.loc[is_num] = dt.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    if (~is_num).any():
        dt_str = pd.to_datetime(series[~is_num], utc=True, errors="coerce")
        out.loc[~is_num] = dt_str.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
    return out


def slugify_topic(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")[:60]


def col_or_default(df: pd.DataFrame, col: str, default) -> pd.Series:
    """Return DataFrame column as a Series or a default-filled Series if missing."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def load_youtube() -> pd.DataFrame:
    if not YOUTUBE_RAW.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(YOUTUBE_RAW, encoding="utf-8-sig")
    out = pd.DataFrame()
    out["platform"] = col_or_default(df, "platform", "youtube").astype(str).str.lower()
    out["source_name"] = col_or_default(df, "source_name", "YouTube").astype(str)
    out["topic_tag"] = col_or_default(df, "topic_tag", "").astype(str)
    out["source_id"] = col_or_default(df, "source_id", "").astype(str)
    out["comment_id"] = col_or_default(df, "comment_id", "").astype(str)
    out["user_hash"] = col_or_default(df, "user_hash", "unknown").astype(str)
    out["timestamp_utc"] = format_epoch_utc(col_or_default(df, "timestamp_utc", ""))
    out["likes"] = pd.to_numeric(col_or_default(df, "likes", 0), errors="coerce").fillna(0).astype(int)
    out["text"] = col_or_default(df, "text", "").map(scrub_pii_text)
    return out[CANONICAL_COLUMNS]


def load_reddit() -> pd.DataFrame:
    if not REDDIT_RAW.exists():
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    df = pd.read_csv(REDDIT_RAW, encoding="utf-8-sig")
    out = pd.DataFrame()
    out["platform"] = col_or_default(df, "platform", "reddit").astype(str).str.lower()
    out["source_name"] = col_or_default(df, "source_name", "r/Philippines").astype(str)
    if "topic_tag" in df.columns:
        out["topic_tag"] = col_or_default(df, "topic_tag", "").astype(str)
    else:
        out["topic_tag"] = col_or_default(df, "source_title", "").map(slugify_topic)
    out["source_id"] = col_or_default(df, "source_id", "").astype(str)
    out["comment_id"] = col_or_default(df, "comment_id", "").astype(str)
    user_col = "author_hash" if "author_hash" in df.columns else "user_hash"
    out["user_hash"] = col_or_default(df, user_col, "anon_unknown").astype(str)
    ts_col = "created_utc" if "created_utc" in df.columns else "timestamp_utc"
    out["timestamp_utc"] = format_epoch_utc(col_or_default(df, ts_col, ""))
    like_col = "score" if "score" in df.columns else "likes"
    out["likes"] = pd.to_numeric(col_or_default(df, like_col, 0), errors="coerce").fillna(0).astype(int)
    out["text"] = col_or_default(df, "text", "").map(scrub_pii_text)
    return out[CANONICAL_COLUMNS]


def load_external() -> list[pd.DataFrame]:
    frames = []
    if not EXTERNAL_DIR.exists():
        return frames
    for csv_path in sorted(EXTERNAL_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            if "text" not in df.columns:
                print(f"[external] Skipping {csv_path.name}: missing 'text' column")
                continue
            out = pd.DataFrame()
            out["platform"] = col_or_default(df, "platform", csv_path.stem).astype(str).str.lower()
            out["source_name"] = col_or_default(df, "source_name", csv_path.stem).astype(str)
            out["topic_tag"] = col_or_default(df, "topic_tag", csv_path.stem).astype(str)
            out["source_id"] = col_or_default(df, "source_id", csv_path.stem).astype(str)
            if "comment_id" in df.columns:
                out["comment_id"] = col_or_default(df, "comment_id", "").astype(str)
            else:
                out["comment_id"] = [f"{csv_path.stem}_{i}" for i in range(len(df))]
            ucol = "user_hash" if "user_hash" in df.columns else "author_hash"
            out["user_hash"] = col_or_default(df, ucol, "anon_unknown").astype(str)
            ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "created_utc"
            if ts_col in df.columns:
                out["timestamp_utc"] = format_epoch_utc(df[ts_col])
            else:
                out["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            lcol = "likes" if "likes" in df.columns else "score"
            out["likes"] = pd.to_numeric(col_or_default(df, lcol, 0), errors="coerce").fillna(0).astype(int)
            out["text"] = col_or_default(df, "text", "").map(scrub_pii_text)
            frames.append(out[CANONICAL_COLUMNS])
            print(f"[external] Loaded {len(out)} rows from {csv_path.name}")
        except Exception as exc:
            print(f"[external] Failed to load {csv_path.name}: {exc}")
    return frames


def main():
    frames = [load_youtube(), load_reddit()] + load_external()
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No raw datasets found to merge.")

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[merged["text"].str.strip() != ""].copy()
    merged = merged.drop_duplicates(subset=["platform", "comment_id"], keep="first").reset_index(drop=True)

    ALL_RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(ALL_RAW_OUT, index=False, encoding="utf-8-sig")

    DELIVERABLE_RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    merged[DELIVERABLE_COLUMNS].to_csv(DELIVERABLE_RAW_OUT, index=False, encoding="utf-8-sig")

    print("\n--- Merged Raw Dataset Summary ---")
    print(f"Total merged raw comments: {len(merged):,}")
    print("\nBy platform:")
    print(merged["platform"].value_counts().to_string())
    print("\nBy top source_name:")
    print(merged["source_name"].value_counts().head(15).to_string())
    print(f"\nCanonical output:   {ALL_RAW_OUT}")
    print(f"Rubric deliverable: {DELIVERABLE_RAW_OUT} (columns: {DELIVERABLE_COLUMNS})")


if __name__ == "__main__":
    main()
