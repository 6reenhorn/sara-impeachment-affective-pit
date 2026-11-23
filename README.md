# Public Affective Analysis of VP Sara Duterte's Impeachment (YouTube & Reddit)

**Course:** CS412 — Affective Computing (PIT)  
**Topic:** Public Sentiment & Emotion Mining on Vice President Sara Duterte's Impeachment Trial (Philippines)  
**Platforms:** YouTube (PH news channels) and Reddit (r/Philippines megathread)  
**Current Status:** Stages 1 (Text Mining) and 2 (Preprocessing) complete for YouTube; Reddit mining complete and ready for preprocessing; Stage 3 (Modeling & Analysis) pending final project rubric.

---

## Dataset Summary

| Platform | Sources | Comment Records | Unique Commenters |
| :--- | :--- | ---: | ---: |
| YouTube | 18 news clips & trial coverages across 8 Philippine news outlets | 5,717 raw → 4,938 clean | 3,740 |
| Reddit | 1 public megathread on r/Philippines | 1,346 (after cross-sort deduplication) | 371 |
| **Total** | **19 sources** | **6,284 usable records** | **4,111 (platform totals; cross-platform overlap not measurable)** |

- **YouTube collection date:** November 20, 2026
- **Reddit thread activity window:** July 6–7, 2026 (UTC); captured from public thread JSON snapshots
- **Anonymization:** All usernames are replaced with one-way SHA-256 hashes. No plain-text usernames are stored.

### Preprocessing & Cleaning Funnel (YouTube)

| Stage | Remaining Comments | Removed |
| :--- | ---: | ---: |
| 0. Raw scraped input (`youtube_comments_raw.csv`) | 5,717 | — |
| 1. Remove empty / URL-only rows | 5,610 | -107 |
| 2. Exact-text deduplication (copy-paste spam removal) | 5,475 | -135 |
| 3. Keep comments with $\ge$ 3 words | 5,198 | -277 |
| 4. Cap at $\le$ 5 comments per unique user hash | **4,938** | -260 |

### Clean YouTube Dataset Distribution by Channel

| News Channel | Clean Comments | Share |
| :--- | ---: | ---: |
| ABS-CBN News | 1,600 | 32.4% |
| UNTV News and Rescue | 1,482 | 30.0% |
| GMA News | 1,051 | 21.3% |
| NET25 News and Information | 273 | 5.5% |
| TV5 News Philippines | 177 | 3.6% |
| Rappler | 158 | 3.2% |
| Bilyonaryo News Channel | 146 | 3.0% |
| Philstar News | 51 | 1.0% |
| **Total** | **4,938** | **100.0%** |

### Reddit Dataset Notes

- Source thread: [r/Philippines — Megathread: Impeachment Trial of Vice President Sara Duterte](https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/)
- Collected from five sort-order snapshots (`top`, `new`, `controversial`, `old`, `qa`) of the same thread.
- The parser removes deleted/removed comments, comments shorter than 3 characters, and duplicates that appear across sort orders (normalized-text comparison).
- Final output: **1,346 comment records** from **371 anonymized commenters**.

---

## Repository Structure

```text
sara-impeachment-affective-pit/
├── data/
│   ├── raw/
│   │   ├── youtube_comments_raw.csv        # Anonymized YouTube scrape
│   │   ├── reddit_comments_raw.csv         # Anonymized Reddit parse output
│   │   └── reddit_json/                    # Local-only saved Reddit JSON (gitignored)
│   └── processed/
│       └── youtube_comments_clean.csv      # Cleaned YouTube dataset ready for modeling
├── src/
│   ├── 01_mining/
│   │   ├── scrape_youtube.py               # YouTube comment scraper
│   │   ├── sources.csv                     # YouTube video list
│   │   └── parse_reddit_json.py            # Reddit saved-JSON parser
│   ├── 02_preprocessing/
│   │   └── clean_comments.py               # YouTube deduplication, user capping, dual-text cleaning
│   └── 03_modeling/                        # Feature extraction & affective classification (upcoming)
├── notebooks/                              # Exploratory analysis and modeling notebooks (upcoming)
├── requirements.txt
└── README.md
```

---

## How to Reproduce

### 0. Install dependencies

Run from the repository root:

```bash
pip install -r requirements.txt
```

### 1a. Mine YouTube comments

Edit `src/01_mining/sources.csv` if needed, then run:

```bash
python src/01_mining/scrape_youtube.py
```

The scraper reads every URL in `sources.csv`, collects comments, deduplicates by comment ID, and writes `data/raw/youtube_comments_raw.csv`.

### 1b. Mine Reddit comments

Reddit data is collected from browser-saved JSON snapshots of the public thread, so no API keys or login are required.

1. Create the local input folder:

   ```powershell
   New-Item -ItemType Directory -Force data/raw/reddit_json
   ```

2. Open each link below in a browser, press **Ctrl+S**, and save it into `data/raw/reddit_json/` with the suggested filename.

   | Sort | Link | Save as |
   | :--- | :--- | :--- |
   | top | `https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/.json?limit=500&sort=top&raw_json=1` | `top.json` |
   | new | `https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/.json?limit=500&sort=new&raw_json=1` | `new.json` |
   | controversial | `https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/.json?limit=500&sort=controversial&raw_json=1` | `controversial.json` |
   | old | `https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/.json?limit=500&sort=old&raw_json=1` | `old.json` |
   | qa | `https://www.reddit.com/r/Philippines/comments/1uophc7/megathread_impeachment_trial_of_vice_president/.json?limit=500&sort=qa&raw_json=1` | `qa.json` |

3. Run the parser:

   ```bash
   python src/01_mining/parse_reddit_json.py
   ```

   It reads every `.json` file in `data/raw/reddit_json/`, removes junk and duplicate comments, hashes usernames, and writes `data/raw/reddit_comments_raw.csv`.

> Raw Reddit JSON snapshots are intentionally **not committed** because they contain public comments tied to usernames. Only the anonymized CSV is tracked.

### 2. Preprocess comments

```bash
python src/02_preprocessing/clean_comments.py
```

This currently applies the YouTube cleaning pipeline (empty/URL removal, exact-text deduplication, minimum word count, per-user cap, affect metadata) and writes `data/processed/youtube_comments_clean.csv`.

---

## Methodology Notes

- **Anonymization:** YouTube channel IDs/authors and Reddit authors are hashed with SHA-256 and shortened; plain usernames are never stored.
- **Reddit deduplication:** The same comment appears in multiple sort-order snapshots, so duplicates are removed by comparing normalized text.
- **Text variants:** Cleaning produces `text_clean` (preserves emojis, capitalization, and punctuation for affect analysis) and `text_tfidf` (normalized lowercase tokens for TF-IDF / classical ML).
- **Limitations:** The corpus covers 18 YouTube news clips and one Reddit megathread. Results reflect these communities and are not representative of the entire Philippine public.

---

## Team

- John Cyril Espina
