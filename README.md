# Public Affective Analysis of VP Sara Duterte's Impeachment (YouTube PH)

**Course:** CS412 — Affective Computing (PIT)  
**Topic:** Public Sentiment & Emotion Mining on Vice President Sara Duterte's Impeachment Trial (Philippines)  
**Current Status:** Stages 1 (Text Mining) and 2 (Preprocessing) complete; Stage 3 (Modeling & Analysis) pending final project rubric.

---

## Dataset Summary

- **Collection Date:** November 20, 2026
- **Primary Platform:** YouTube (18 news clips & trial coverages across 8 Philippine news outlets)
- **Raw Comments Collected:** 5,717
- **Final Cleaned Comments:** 4,938 (86.4% retention; minimum requirement: 1,000)
- **Unique Commenters (Anonymized):** 3,740
- **Median Comment Length:** 12.0 words

### Preprocessing & Cleaning Funnel

| Stage | Remaining Comments | Removed |
| :--- | ---: | ---: |
| 0. Raw scraped input (`youtube_comments_raw.csv`) | 5,717 | — |
| 1. Remove empty / URL-only rows | 5,610 | -107 |
| 2. Exact-text deduplication (copy-paste spam removal) | 5,475 | -135 |
| 3. Keep comments with $\ge$ 3 words | 5,198 | -277 |
| 4. Cap at $\le$ 5 comments per unique user hash | **4,938** | -260 |

### Clean Dataset Distribution by Channel

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

---

## Repository Structure

```text
sara-impeachment-affective-pit/
├── data/
│   ├── raw/                  # Anonymized raw YouTube scrape (youtube_comments_raw.csv)
│   ├── external/             # Reserved for additional platform exports (e.g., Reddit CSVs)
│   └── processed/            # Cleaned dataset ready for modeling (youtube_comments_clean.csv)
├── src/
│   ├── 01_mining/            # YouTube scraper + sources.csv video list
│   ├── 02_preprocessing/     # Deduplication, user capping, and dual-text cleaning
│   └── 03_modeling/          # Feature extraction & affective classification (upcoming)
├── notebooks/                # Exploratory analysis and modeling notebooks (upcoming)
├── requirements.txt
└── README.md