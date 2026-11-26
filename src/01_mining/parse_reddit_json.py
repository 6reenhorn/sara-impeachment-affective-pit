import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd

JSON_FOLDER = Path("data/raw/reddit_json")
OUTPUT_FILE = Path("data/raw/reddit_comments_raw.csv")
TARGET_TOTAL = 1500

COLUMNS = [
    "platform",
    "source_name",
    "topic_tag",
    "source_id",
    "source_title",
    "comment_id",
    "author_hash",
    "text",
    "score",
    "created_utc",
]

USER_MENTION_RE = re.compile(r"(?:^|(?<=\s))/?u/[A-Za-z0-9_-]+", re.IGNORECASE)


def anonymize(name):
    if not name or str(name) in ("None", "[deleted]", "nan"):
        return "anon_unknown"
    return "anon_" + hashlib.sha256(str(name).encode()).hexdigest()[:12]


def clean_text(t):
    unescaped = html.unescape(str(t or ""))
    scrubbed = USER_MENTION_RE.sub(" ", unescaped)
    return re.sub(r"\s+", " ", scrubbed).strip()


def slugify_topic(title: str, post_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(title or "").lower()).strip("_")
    return slug[:60] or str(post_id)


def is_junk(t):
    t = t.strip().lower()
    return len(t) < 3 or t in ("[deleted]", "[removed]")


def walk(children, out, post_id, title, subreddit_name, topic_tag):
    for ch in children:
        if ch.get("kind") != "t1":
            continue
        d = ch["data"]
        text = clean_text(d.get("body"))
        if not is_junk(text):
            out.append({
                "platform": "reddit",
                "source_name": subreddit_name,
                "topic_tag": topic_tag,
                "source_id": post_id,
                "source_title": title,
                "comment_id": d.get("id"),
                "author_hash": anonymize(d.get("author")),
                "text": text,
                "score": d.get("score"),
                "created_utc": d.get("created_utc"),
            })
        replies = d.get("replies")
        if isinstance(replies, dict):
            walk(
                replies["data"]["children"],
                out,
                post_id,
                title,
                subreddit_name,
                topic_tag,
            )


def load_json_files():
    folder = Path(JSON_FOLDER)
    files = sorted(folder.rglob("*.json")) if folder.exists() else []
    if not files:
        print(
            f"No .json files found in {folder}. "
            f"Keeping existing {OUTPUT_FILE} untouched if present."
        )
        return []
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
            post = data[0]["data"]["children"][0]["data"]
            subreddit = post.get("subreddit_name_prefixed") or f"r/{post.get('subreddit', 'Philippines')}"
            topic_tag = slugify_topic(post.get("title", ""), post["id"])
            before = len(rows)
            walk(
                data[1]["data"]["children"],
                rows,
                post["id"],
                post["title"],
                subreddit,
                topic_tag,
            )
            print(f"[json] {f.name}: +{len(rows) - before} comments ({subreddit} | {post['id']})")
        except Exception as e:
            print(f"[json] could not read {f.name}: {e}")
    return rows


def main():
    rows = load_json_files()
    if not rows:
        print("\nNothing to re-parse. Add .json files under data/raw/reddit_json/ to update.")
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[~df["text"].map(is_junk)].copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values(by=["score", "created_utc"], ascending=[False, True])

    before_id = len(df)
    df = df.drop_duplicates(subset=["comment_id"], keep="first")
    print(f"Removed {before_id - len(df)} duplicate comment_ids")

    df["_k"] = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    before_txt = len(df)
    df = df.drop_duplicates("_k").drop(columns="_k").reset_index(drop=True)
    print(f"Removed {before_txt - len(df)} duplicate texts")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nTotal unique comments: {len(df)} (target {TARGET_TOTAL})")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
