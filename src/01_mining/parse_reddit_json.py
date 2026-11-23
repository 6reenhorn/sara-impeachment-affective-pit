import hashlib
import json
import re
from pathlib import Path

import pandas as pd

JSON_FOLDER = Path("data/raw/reddit_json")
OUTPUT_FILE = Path("data/raw/reddit_comments_raw.csv")
TARGET_TOTAL = 1000

COLUMNS = ["platform", "source_id", "source_title", "comment_id",
        "author_hash", "text", "score", "created_utc"]


def anonymize(name):
    if not name or str(name) in ("None", "[deleted]", "nan"):
        return "anon_unknown"
    return "anon_" + hashlib.sha256(str(name).encode()).hexdigest()[:10]


def clean_text(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def is_junk(t):
    t = t.strip().lower()
    return len(t) < 3 or t in ("[deleted]", "[removed]")


def walk(children, out, post_id, title):
    for ch in children:
        if ch.get("kind") != "t1":
            continue
        d = ch["data"]
        text = clean_text(d.get("body"))
        if not is_junk(text):
            out.append({
                "platform": "reddit", "source_id": post_id,
                "source_title": title, "comment_id": d.get("id"),
                "author_hash": anonymize(d.get("author")), "text": text,
                "score": d.get("score"), "created_utc": d.get("created_utc"),
            })
        replies = d.get("replies")
        if isinstance(replies, dict):
            walk(replies["data"]["children"], out, post_id, title)


def load_json_files():
    folder = Path(JSON_FOLDER)
    files = sorted(folder.glob("*.json")) if folder.exists() else []
    if not files:
        print(f"No .json files found. Looked in:\n  {folder}\n"
              f"Create that folder and save the thread .json files in it.")
        return []
    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
            post = data[0]["data"]["children"][0]["data"]
            before = len(rows)
            walk(data[1]["data"]["children"], rows, post["id"], post["title"])
            print(f"[json] {f.name}: +{len(rows) - before} comments")
        except Exception as e:
            print(f"[json] could not read {f.name}: {e}")
    return rows


def main():
    rows = load_json_files()
    if not rows:
        print("\nNothing to process yet. Add the .json files, then run again.")
        return
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df[~df["text"].map(is_junk)]
    df["_k"] = df["text"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    before = len(df)
    df = df.drop_duplicates("_k").drop(columns="_k")
    print(f"Removed {before - len(df)} duplicates")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nTotal unique comments: {len(df)} (target {TARGET_TOTAL})")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
