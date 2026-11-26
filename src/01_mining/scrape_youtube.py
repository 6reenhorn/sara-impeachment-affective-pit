import hashlib
import time
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
from youtube_comment_downloader import (
    SORT_BY_POPULAR,
    SORT_BY_RECENT,
    YoutubeCommentDownloader,
)

SOURCES_FILE = Path("src/01_mining/sources.csv")
OUTPUT_FILE = Path("data/raw/youtube_comments_raw.csv")


def extract_video_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/").split("/")[0]
    query_v = parse_qs(parsed.query).get("v")
    if query_v:
        return query_v[0]
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) >= 2 and parts[0] in {"live", "shorts", "embed"}:
        return parts[1]
    return url.strip()


def anonymize_user(channel_id: str, author: str) -> str:
    raw_id = (channel_id or author or "").strip()
    if not raw_id:
        return "unknown"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:12]


def parse_votes(raw_votes) -> int:
    if raw_votes is None:
        return 0
    s = str(raw_votes).strip().upper().replace(",", "")
    if not s:
        return 0
    try:
        if s.endswith("K"):
            return int(float(s[:-1]) * 1_000)
        if s.endswith("M"):
            return int(float(s[:-1]) * 1_000_000)
        return int(float(s))
    except ValueError:
        return 0


def format_timestamp(time_parsed) -> str:
    if not time_parsed:
        return ""
    try:
        return datetime.fromtimestamp(float(time_parsed), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    except (TypeError, ValueError, OSError):
        return ""


def scrape_video(downloader, row) -> list[dict]:
    url = str(row["url"]).strip()
    video_id = extract_video_id(url)
    limit = int(row.get("limit", 500))
    sort_choice = str(row.get("sort_by", "popular")).strip().lower()
    sort_mode = SORT_BY_RECENT if sort_choice == "recent" else SORT_BY_POPULAR

    print(f"Mining [{row['channel']}] {video_id} (limit={limit}, sort={sort_choice})...")
    comments = downloader.get_comments_from_url(url, sort_by=sort_mode)

    collected = []
    for c in islice(comments, limit):
        text = (c.get("text") or "").strip()
        if not text:
            continue

        collected.append(
            {
                "platform": "youtube",
                "source_name": row["channel"],
                "topic_tag": row.get("topic_tag", ""),
                "source_id": video_id,
                "comment_id": c.get("cid", ""),
                "user_hash": anonymize_user(c.get("channel", ""), c.get("author", "")),
                "timestamp_utc": format_timestamp(c.get("time_parsed")),
                "relative_time": c.get("time", ""),
                "likes": parse_votes(c.get("votes")),
                "is_reply": bool(c.get("reply", False)),
                "text": text,
                "collected_at_utc": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )

    print(f"  -> Collected {len(collected)} comments.")
    return collected


def main():
    if not SOURCES_FILE.exists():
        raise FileNotFoundError(f"Missing {SOURCES_FILE}. Add your YouTube links first.")

    sources = pd.read_csv(SOURCES_FILE)
    downloader = YoutubeCommentDownloader()
    all_rows = []

    for _, row in sources.iterrows():
        try:
            all_rows.extend(scrape_video(downloader, row))
        except Exception as exc:
            print(f"  !! Skipped {row['url']} due to error: {exc}")
        time.sleep(2)  # Polite pause between videos

    if not all_rows and not OUTPUT_FILE.exists():
        raise RuntimeError("No comments were collected. Check your URLs and connection.")

    new_df = pd.DataFrame(all_rows)
    if OUTPUT_FILE.exists():
        existing_df = pd.read_csv(OUTPUT_FILE, encoding="utf-8-sig")
        print(f"Loaded {len(existing_df)} existing rows from {OUTPUT_FILE} for incremental merge.")
        # Put newly scraped rows first so fresher counts/likes win on comment_id dedup
        df = pd.concat([new_df, existing_df], ignore_index=True)
    else:
        df = new_df

    raw_count = len(df)
    df = df.drop_duplicates(subset=["comment_id"], keep="first").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig keeps Tagalog/Bisaya characters and emojis intact when opened in Excel
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    unique_texts = df["text"].str.lower().str.strip().nunique()
    print("\n--- Mining Summary ---")
    print(f"Raw collected:      {raw_count}")
    print(f"Unique comment IDs: {len(df)}")
    print(f"Unique text bodies: {unique_texts}")
    print(f"Unique users (hash):{df['user_hash'].nunique()}")
    print("\nBy channel:")
    print(df["source_name"].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()