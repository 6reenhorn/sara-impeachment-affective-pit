# Public Affective Analysis of VP Sara Duterte's Impeachment (YouTube & Reddit)

**Course:** CS412 — CS Elective 2 (Affective Computing, 1st Sem SY 2026–2027)  
**Instructor:** Love Jhoye M. Raboy, PhD  
**Topic:** Public Sentiment & Emotion Mining on Vice President Sara Duterte's Impeachment Trial (Philippines)  
**Platforms:** YouTube (10 Philippine news outlets, 33 videos) and Reddit (`r/Philippines`)  
**Current Status:** End-to-end Stages 1–5 implemented (`14,110` raw → `11,969` clean comments). Lexicon & comparative pipelines complete; ML models currently use the **Provisional Consensus Baseline** while the **5-member Human Gold Standard** sheets (`data/annotated/sheets/member_1.csv` .. `member_5.csv`) await team annotation.

---

## Dataset Summary

| Platform | Sources | Raw Merged Records | Final Clean Comments | Unique Commenters (Clean) |
| :--- | :--- | ---: | ---: | ---: |
| YouTube | 33 news clips & trial coverages across 10 Philippine news outlets | 12,779 | 11,102 | 8,405 |
| Reddit | Public megathread on `r/Philippines` (expandable with additional threads) | 1,331 | 867 | 362 |
| **Total** | **34 sources** | **14,110** | **11,969** | **8,767** |

- **YouTube collection:** Expanded November 25, 2026 across 10 outlets (`ABS-CBN News`, `UNTV News and Rescue`, `TV5 News Philippines`, `GMA News`, `NET25 News and Information`, `Rappler`, `Bilyonaryo News Channel`, `Philstar News`, `INQUIRER.net`, `ANC 24/7`).
- **Reddit collection:** Public `r/Philippines` thread JSON snapshots (`1uophc7`), July 6–7, 2026 UTC.
- **Ethical & Privacy Compliance:** Only public posts/comments are collected, with zero interaction. Usernames/author IDs are one-way SHA-256 hashed for anti-spam deduplication, and inline `@handle` / `u/handle` mentions + HTML entities are scrubbed from comment bodies. The rubric deliverable `deliverables/raw_comments.csv` contains only `platform`, `timestamp_utc`, `likes`, and `text`.

### Unified Preprocessing & Cleaning Funnel

| Stage | Total | YouTube | Reddit |
| :--- | ---: | ---: | ---: |
| 0. Raw merged & PII-scrubbed input (`all_comments_raw.csv`) | 14,110 | 12,779 | 1,331 |
| 1. Clean HTML/URLs/handles & drop empty/URL-only | 13,888 | 12,560 | 1,328 |
| 2. Exact normalized-text deduplication | 13,573 | 12,245 | 1,328 |
| 3. Keep comments with $\ge$ 3 words | 12,903 | 11,641 | 1,262 |
| 4. Cap at $\le$ 5 comments per unique `(platform, user_hash)` | **11,969** | **11,102** | **867** |
| 5. Tokenize + EN/FIL stopword removal (negations kept) + POS lemmatization | **11,969** | **11,102** | **867** |

### Clean Dataset Distribution by Source

| Source | Clean Comments | Share of Total |
| :--- | ---: | ---: |
| ABS-CBN News (YouTube) | 2,993 | 25.0% |
| UNTV News and Rescue (YouTube) | 2,819 | 23.6% |
| TV5 News Philippines (YouTube) | 2,510 | 21.0% |
| GMA News (YouTube) | 1,720 | 14.4% |
| r/Philippines (Reddit) | 867 | 7.2% |
| NET25 News and Information (YouTube) | 570 | 4.8% |
| Rappler (YouTube) | 208 | 1.7% |
| Bilyonaryo News Channel (YouTube) | 149 | 1.2% |
| Philstar News (YouTube) | 64 | 0.5% |
| INQUIRER.net (YouTube) | 39 | 0.3% |
| ANC 24/7 (YouTube) | 30 | 0.3% |
| **Total** | **11,969** | **100.0%** |

---

## Key Rubric Deliverables (`deliverables/`)

| Rubric Task | Deliverable File(s) | Description |
| :--- | :--- | :--- |
| **Task 1: Data Collection** | `deliverables/raw_comments.csv` | 14,110 raw anonymized comments (`platform`, `timestamp_utc`, `likes`, `text` only) |
| **Task 2: Preprocessing** | `deliverables/cleaned_comments.csv`<br>`deliverables/preprocessing_summary.txt` | 11,969 cleaned comments with `text_clean`, `tokens`, `tokens_no_stop`, `text_lemmatized` |
| **Task 3: Lexicon Sentiment** | `deliverables/lexicon_results.csv`<br>`deliverables/charts/lexicon_*.png` | VADER, TextBlob, and AFINN polarity scores, labels, agreement matrix & language diagnostics |
| **Task 4: ML Sentiment** | `deliverables/ml_results.txt`<br>`deliverables/confusion_matrix.png` | TF-IDF (1–2 grams) + Logistic Regression & Multinomial Naive Bayes metrics + confusion matrices |
| **Task 5 & 6: Analysis & Slides** | `deliverables/comparative_analysis.md`<br>`deliverables/charts/*.png` | Task 5 Q&A, 9-slide Task 6 outline, disagreement case studies, Page 5 Lab Form notes |

---

## Repository Structure

```text
sara-impeachment-affective-pit/
├── data/
│   ├── raw/                        # youtube_comments_raw.csv, reddit_comments_raw.csv, all_comments_raw.csv
│   ├── external/                   # Optional additional CSV exports from teammates
│   ├── processed/                  # all_comments_clean.csv, youtube_comments_clean.csv, full_corpus_predictions.csv
│   └── annotated/
│       ├── ANNOTATION_GUIDELINES.md
│       ├── sample_index.csv
│       └── sheets/                 # member_1.csv .. member_5.csv (250 rows each: 50 shared + 200 unique)
├── deliverables/                   # Exact instructor-required outputs + 300 DPI charts
├── src/
│   ├── 01_mining/                  # scrape_youtube.py, sources.csv, parse_reddit_json.py, merge_raw_datasets.py
│   ├── 02_preprocessing/           # clean_comments.py (cleaning, tokenization, EN/FIL stopwords, lemmatization)
│   ├── 03_lexicon/                 # run_lexicons.py (VADER, TextBlob, AFINN)
│   ├── 04_modeling/                # prepare_annotation_sheets.py, merge_annotations.py, train_ml_models.py
│   └── 05_analysis/                # comparative_analysis.py
├── tests/
│   └── verify_deliverables.py      # Automated ethics, schema, NLP, ML, and annotation-sheet checks
├── run_pipeline.py                 # Single-command runner for Stages 1b -> 5
└── requirements.txt
```

---

## How to Reproduce & Complete Team Workflows

### 0. Install Dependencies

```bash
pip install -r requirements.txt
```

### 1. Run the Full Pipeline

```bash
python run_pipeline.py
```

### 2. Optional: Add More `r/Philippines` Reddit Threads

Because Reddit blocks automated Python scripts with HTTP 403, save public thread JSONs manually in your browser (`Ctrl+S`) into `data/raw/reddit_json/` (multiple threads and sorts supported), then run:

```bash
python src/01_mining/parse_reddit_json.py
python run_pipeline.py
```

### 3. Complete the 5-Member Human Gold Standard Annotation (Recommended Final Step)

1. Each of the 5 members opens their assigned file in `data/annotated/sheets/` (`member_1.csv` .. `member_5.csv`) and reads `data/annotated/ANNOTATION_GUIDELINES.md`.
2. Fill the `human_label` column (`Positive`, `Negative`, or `Neutral`) for all 250 rows (50 shared calibration rows for Fleiss' $\kappa$ + 200 unique rows = 1,050 unique gold-labeled comments).
3. Merge labels, compute inter-annotator agreement, and re-run modeling + analysis on the true Human Gold Standard:

```bash
python src/04_modeling/merge_annotations.py
python run_pipeline.py --mode gold
pytest -v tests/verify_deliverables.py
```

---

## Team (5 Members)

- John Cyril Espina — `data/annotated/sheets/member_1.csv` *(completed: 250/250)*
- Den Jester Antonio — `data/annotated/sheets/member_2.csv` *(completed: 250/250)*
- *(Teammate 3 — Data collection & `member_3.csv` annotation)*
- *(Teammate 4 — Data collection & `member_4.csv` annotation)*
- *(Teammate 5 — Data collection & `member_5.csv` annotation)*
