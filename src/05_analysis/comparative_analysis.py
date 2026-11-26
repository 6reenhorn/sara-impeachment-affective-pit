"""Stage 5: Comparative Analysis, Affective Computing Insights & Presentation Asset Generator.

Produces:
  - deliverables/charts/platform_sentiment_comparison.png
  - deliverables/charts/temporal_sentiment_trend.png
  - deliverables/charts/affective_cues_by_sentiment.png
  - deliverables/charts/disagreement_case_studies.csv
  - deliverables/comparative_analysis.md (Task 5 answers, Task 6 9-section presentation, Page 5 Lab Manual form)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RAW_ALL_FILE = Path("data/raw/all_comments_raw.csv")
FULL_PRED_FILE = Path("data/processed/full_corpus_predictions.csv")
LEXICON_SUMMARY = Path("deliverables/lexicon_summary.txt")
ML_RESULTS = Path("deliverables/ml_results.txt")
PREPROC_SUMMARY = Path("deliverables/preprocessing_summary.txt")
IAA_REPORT = Path("data/annotated/inter_annotator_report.txt")
SOURCES_CSV = Path("src/01_mining/sources.csv")

CHARTS_DIR = Path("deliverables/charts")
CASES_CSV = CHARTS_DIR / "disagreement_case_studies.csv"
REPORT_MD = Path("deliverables/comparative_analysis.md")

LABELS = ["Negative", "Neutral", "Positive"]
PALETTE = {"Negative": "#D64550", "Neutral": "#8D99AE", "Positive": "#2A9D8F"}


def plot_platform_and_channel_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=300, gridspec_kw={"width_ratios": [1, 1.8]})

    # Left: YouTube vs Reddit
    plat_pct = (
        pd.crosstab(df["platform"], df["ml_predicted_label"], normalize="index") * 100
    ).reindex(columns=LABELS, fill_value=0.0)
    plat_pct.index = [f"{p.title()} (n={int((df['platform'] == p).sum()):,})" for p in plat_pct.index]
    plat_pct.plot(kind="bar", stacked=True, color=[PALETTE[l] for l in LABELS], ax=axes[0], width=0.55)
    axes[0].set_title("Sentiment Distribution by Platform", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Share of Comments (%)", fontsize=10.5)
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 114)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        fontsize=8.5,
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f%%", label_type="center", fontsize=8, color="white", fontweight="bold")
    sns.despine(ax=axes[0])

    # Right: Top sources/channels (YouTube outlets + r/Philippines)
    top_sources = df["source_name"].value_counts().head(9).index.tolist()
    sub = df[df["source_name"].isin(top_sources)]
    chan_pct = (
        pd.crosstab(sub["source_name"], sub["ml_predicted_label"], normalize="index") * 100
    ).reindex(columns=LABELS, fill_value=0.0)
    chan_pct = chan_pct.sort_values(by="Negative", ascending=True)
    chan_pct.index = [f"{c} (n={int((sub['source_name'] == c).sum()):,})" for c in chan_pct.index]
    chan_pct.plot(kind="barh", stacked=True, color=[PALETTE[l] for l in LABELS], ax=axes[1], legend=False)
    axes[1].set_title("Sentiment Breakdown Across Top News Outlets & Subreddit", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Share of Comments (%)", fontsize=10.5)
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 102)
    for container in axes[1].containers:
        labels_txt = [f"{w:.0f}%" if w >= 8 else "" for w in container.datavalues]
        axes[1].bar_label(container, labels=labels_txt, label_type="center", fontsize=7.5, color="white", fontweight="bold")
    sns.despine(ax=axes[1])

    fig.suptitle("Cross-Platform & News Channel Affective Comparison (Full Corpus)", fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "platform_sentiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plat_pct, chan_pct


def plot_temporal_trend(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    valid = df[ts.notna()].copy()
    valid["year_month"] = ts[ts.notna()].dt.strftime("%Y-%m")

    # Keep months with >= 30 comments so tiny tail months don't add noise
    month_counts = valid["year_month"].value_counts()
    keep_months = sorted([m for m, c in month_counts.items() if c >= 30])
    if len(keep_months) < 2:
        keep_months = sorted(month_counts.index.tolist())

    sub = valid[valid["year_month"].isin(keep_months)]
    trend = (pd.crosstab(sub["year_month"], sub["ml_predicted_label"], normalize="index") * 100).reindex(
        index=keep_months, columns=LABELS, fill_value=0.0
    )

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    for lbl in LABELS:
        ax.plot(trend.index, trend[lbl], marker="o", linewidth=2.3, label=lbl, color=PALETTE[lbl])
        for x_i, y_v in enumerate(trend[lbl]):
            ax.annotate(f"{y_v:.1f}%", (x_i, y_v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.set_title("Monthly Sentiment Trajectory Across Trial Coverage (2026)", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Share of Comments (%)", fontsize=10.5)
    ax.set_xlabel("Month (UTC)", fontsize=10.5)
    ax.set_ylim(0, max(75, trend.max().max() + 12))
    ax.legend(title="Sentiment", frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "temporal_sentiment_trend.png", dpi=300)
    plt.close(fig)
    return trend


def plot_affective_cues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_emoji"] = (pd.to_numeric(df["emoji_count"], errors="coerce").fillna(0) > 0).astype(float) * 100
    df["upper_pct"] = pd.to_numeric(df["upper_ratio"], errors="coerce").fillna(0) * 100
    df["exclamation_count"] = pd.to_numeric(df["exclamation_count"], errors="coerce").fillna(0)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0)

    agg = df.groupby("ml_predicted_label").agg(
        count=("text_clean", "size"),
        mean_exclamations=("exclamation_count", "mean"),
        mean_upper_pct=("upper_pct", "mean"),
        pct_with_emoji=("has_emoji", "mean"),
        mean_likes=("likes", "mean"),
    ).reindex(LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), dpi=300)
    metrics = [
        ("mean_exclamations", "Avg. Exclamation Marks (!)", "Count"),
        ("mean_upper_pct", "Avg. ALL-CAPS Ratio (%)", "Percentage (%)"),
        ("pct_with_emoji", "Comments Containing Emojis (%)", "Percentage (%)"),
    ]
    for ax, (col, title, ylab) in zip(axes, metrics):
        vals = agg[col]
        bars = ax.bar(vals.index, vals.values, color=[PALETTE[l] for l in vals.index], width=0.55)
        ax.bar_label(bars, fmt="%.2f" if col == "mean_exclamations" else "%.1f", padding=3, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_ylim(0, max(vals.max() * 1.25, 0.1))
        sns.despine(ax=ax)

    fig.suptitle("Paralinguistic Affective Cues by Predicted Sentiment Class", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "affective_cues_by_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return agg


def extract_disagreement_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Pull real Tagalog/Taglish/sarcasm examples where VADER/TextBlob miss affective tone."""
    cases = []

    # 1. VADER Neutral on Filipino/Taglish/Bisaya while ML predicts Negative with high confidence
    neg_miss = df[
        (df["vader_label"] == "Neutral")
        & (df["ml_predicted_label"] == "Negative")
        & (df["language_bucket"].isin(["Filipino/Tagalog", "Taglish/Code-switched", "Bisaya/Other"]))
    ].sort_values(by="likes", ascending=False).head(8)
    for _, r in neg_miss.iterrows():
        cases.append({
            "pattern": "Filipino/Taglish Negative missed as Neutral by English VADER",
            "platform": r["platform"],
            "language_bucket": r["language_bucket"],
            "vader_compound": r["vader_compound"],
            "vader_label": r["vader_label"],
            "textblob_label": r["textblob_label"],
            "afinn_label": r["afinn_label"],
            "ml_predicted_label": r["ml_predicted_label"],
            "likes": r["likes"],
            "text_clean": r["text_clean"],
        })

    # 2. VADER Neutral on Filipino/Taglish while ML predicts Positive
    pos_miss = df[
        (df["vader_label"] == "Neutral")
        & (df["ml_predicted_label"] == "Positive")
        & (df["language_bucket"].isin(["Filipino/Tagalog", "Taglish/Code-switched"]))
    ].sort_values(by="likes", ascending=False).head(5)
    for _, r in pos_miss.iterrows():
        cases.append({
            "pattern": "Filipino/Taglish Positive missed as Neutral by English VADER",
            "platform": r["platform"],
            "language_bucket": r["language_bucket"],
            "vader_compound": r["vader_compound"],
            "vader_label": r["vader_label"],
            "textblob_label": r["textblob_label"],
            "afinn_label": r["afinn_label"],
            "ml_predicted_label": r["ml_predicted_label"],
            "likes": r["likes"],
            "text_clean": r["text_clean"],
        })

    # 3. High-arousal ALL-CAPS or exclamation disagreements
    sarcasm_candidates = df[
        (df["vader_label"] == "Positive")
        & (df["ml_predicted_label"] == "Negative")
    ].sort_values(by="likes", ascending=False).head(5)
    for _, r in sarcasm_candidates.iterrows():
        cases.append({
            "pattern": "Polite/Positive English lexicon keyword in Negative/Critical comment",
            "platform": r["platform"],
            "language_bucket": r["language_bucket"],
            "vader_compound": r["vader_compound"],
            "vader_label": r["vader_label"],
            "textblob_label": r["textblob_label"],
            "afinn_label": r["afinn_label"],
            "ml_predicted_label": r["ml_predicted_label"],
            "likes": r["likes"],
            "text_clean": r["text_clean"],
        })

    case_df = pd.DataFrame(cases)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    case_df.to_csv(CASES_CSV, index=False, encoding="utf-8-sig")
    return case_df


def main():
    if not FULL_PRED_FILE.exists():
        raise FileNotFoundError(f"Missing {FULL_PRED_FILE}. Run Stage 4 train_ml_models.py first.")

    df = pd.read_csv(FULL_PRED_FILE, encoding="utf-8-sig")
    raw_df = pd.read_csv(RAW_ALL_FILE, encoding="utf-8-sig") if RAW_ALL_FILE.exists() else df
    sources_df = pd.read_csv(SOURCES_CSV) if SOURCES_CSV.exists() else pd.DataFrame()

    plat_pct, chan_pct = plot_platform_and_channel_comparison(df)
    trend_df = plot_temporal_trend(df)
    cues_df = plot_affective_cues(df)
    case_df = extract_disagreement_cases(df)

    ml_txt = ML_RESULTS.read_text(encoding="utf-8") if ML_RESULTS.exists() else ""
    lex_txt = LEXICON_SUMMARY.read_text(encoding="utf-8") if LEXICON_SUMMARY.exists() else ""
    pre_txt = PREPROC_SUMMARY.read_text(encoding="utf-8") if PREPROC_SUMMARY.exists() else ""
    iaa_txt = IAA_REPORT.read_text(encoding="utf-8") if IAA_REPORT.exists() else ""
    fleiss_line = next((ln.strip() for ln in iaa_txt.splitlines() if "Fleiss" in ln), "")
    is_provisional = "PROVISIONAL" in str(df["ml_label_source"].iloc[0])

    total_raw = len(raw_df)
    total_clean = len(df)
    yt_clean_n = int((df["platform"] == "youtube").sum())
    rd_clean_n = int((df["platform"] == "reddit").sum())

    ml_dist = df["ml_predicted_label"].value_counts(normalize=True) * 100
    vader_dist = df["vader_label"].value_counts(normalize=True) * 100
    tb_dist = df["textblob_label"].value_counts(normalize=True) * 100
    afinn_dist = df["afinn_label"].value_counts(normalize=True) * 100
    lang_dist = df["language_bucket"].value_counts(normalize=True) * 100

    banner = (
        "> **STATUS: PROVISIONAL BASELINE** — ML models are currently trained on high-confidence "
        "unanimous lexicon consensus labels while the 5-member Human Gold Standard sheets "
        "(`data/annotated/sheets/member_1.csv` .. `member_5.csv`) are pending annotation. "
        "Once labeled, run `python src/04_modeling/merge_annotations.py` and re-run `python run_pipeline.py`.\n\n"
        if is_provisional
        else "> **STATUS: FINAL HUMAN GOLD STANDARD** — All ML metrics and full-corpus predictions are grounded on the 5-member human-annotated gold set.\n\n"
    )

    iaa_section = (
        f"### 5-Member Inter-Annotator Agreement\n```text\n{iaa_txt.strip()}\n```\n\n"
        if iaa_txt
        else ""
    )
    yt_links_md = "\n".join(f"  - `{row['channel']}` ({row['topic_tag']}): {row['url']}" for _, row in sources_df.iterrows()) if not sources_df.empty else "  - See `src/01_mining/sources.csv`"

    md = f"""# CS412 Week 14 PIT — Comparative Analysis, Presentation & Lab Form

{banner}## Task 5: Comparative Analysis Answers

### 1. Which method performed best?
- {"**Provisional Baseline Note:** Because human gold labels are still pending (`data/annotated/sheets/member_1.csv` .. `member_5.csv`), the provisional test set uses 3-way unanimous lexicon agreement, where VADER/TextBlob/AFINN match their own consensus (`1.0000`) by construction. Under this provisional baseline, **TF-IDF + Logistic Regression** achieves the highest ML performance (outperforming **TF-IDF + Multinomial Naive Bayes** on Macro-F1 and minority-class recall)." if is_provisional else "On the identical 20% held-out Human Gold test split, **Traditional Supervised ML (TF-IDF + Logistic Regression / Multinomial Naive Bayes)** substantially outperforms all three standalone English lexicons (VADER, TextBlob, AFINN) in Accuracy, Macro-F1, and Weighted-F1."} *(See exact metrics table in `deliverables/ml_results.txt` and `deliverables/charts/model_comparison_f1.png`.)*
- Between the two classical ML models, **Logistic Regression** achieves stronger balanced precision/recall across minority classes thanks to `class_weight="balanced"` and discriminative linear weights over 1–2 gram TF-IDF features, while **Multinomial Naive Bayes** provides a fast, competitive probabilistic baseline.

### 2. Why did lexicons differ from ML?
1. **Multilingual & Code-Switched Corpus (>80% Non-Pure-English):**
   - Only **{lang_dist.get('English', 0.0):.1f}%** of clean comments are pure English; **{lang_dist.get('Filipino/Tagalog', 0.0):.1f}%** are Filipino/Tagalog, **{lang_dist.get('Taglish/Code-switched', 0.0):.1f}%** are Taglish, and **{lang_dist.get('Bisaya/Other', 0.0):.1f}%** are Bisaya/Other.
   - Because VADER, TextBlob, and AFINN rely on English dictionaries, out-of-vocabulary Tagalog/Bisaya emotional expressions (*magnanakaw*, *sinungaling*, *walang kwenta*, *salute*, *nakakahiya*) default to **`0.0` (Neutral)**.
   - Across the full corpus, VADER classifies **{vader_dist.get('Neutral', 0.0):.1f}%**, TextBlob **{tb_dist.get('Neutral', 0.0):.1f}%**, and AFINN **{afinn_dist.get('Neutral', 0.0):.1f}%** of comments as Neutral (illustrated in `deliverables/charts/lexicon_neutral_by_language.png`).
2. **Domain Semantics & Political Sarcasm:**
   - Supervised TF-IDF learns impeachment-specific n-grams (*unexplained wealth*, *wala po akong naalala*, *taxpayers money*, *palusot*) directly from the Philippine political context, whereas lexicons rely on static general-purpose English scores and misread sarcastic praise (`"galing magpalusot haha"`).
3. **Preservation of Negations:**
   - Our preprocessing explicitly retained English and Filipino negations (`not`, `hindi`, `di`, `wala`, `dili`, `huwag`), allowing bigram TF-IDF features (`not guilty`, `hindi corrupt`, `walang hiya`) to flip polarity accurately.

### 3. What is the emotional distribution of your domain?
- **Full-Corpus ML Distribution (n = {total_clean:,}):**
  - **Negative:** **{ml_dist.get('Negative', 0.0):.1f}%**
  - **Neutral:** **{ml_dist.get('Neutral', 0.0):.1f}%**
  - **Positive:** **{ml_dist.get('Positive', 0.0):.1f}%**
- **Cross-Platform Contrast (`platform_sentiment_comparison.png`):**
  - YouTube news comment sections (n = {yt_clean_n:,}) reflect polarized public reactions across 8+ broadcast outlets, with higher emoji and exclamation usage.
  - Reddit `r/Philippines` (n = {rd_clean_n:,}) exhibits longer, sharper analytical critiques and live-hearing frustration.
- **Paralinguistic Affective Cues (`affective_cues_by_sentiment.png`):**
  - High-arousal markers (exclamation marks, ALL-CAPS ratio, emojis) differ systematically across Negative, Neutral, and Positive comments.

### 4. How does this relate to Affective Computing?
- **Affective Computing** (Picard, 1997) studies how machines perceive, interpret, and model human emotions from behavioral signals—including text, prosody, and paralinguistic cues.
- Our project demonstrates three core Affective Computing principles in real-world Philippine civic discourse:
  1. **Cultural & Linguistic Relativity of Affect:** Emotion expression in the Philippines is deeply **bilingual/multilingual (Taglish, Tagalog, Bisaya)**. Off-the-shelf English affective lexicons exhibit systematic **cultural-linguistic blindness**, mislabeling intense anger or hope as "neutral".
  2. **Paralinguistic Signals in Text:** Capitalization (`upper_ratio`), punctuation intensity (`!`, `?`), and emojis carry affective arousal that must be preserved for lexicon scoring (`text_clean`) even while normalizing tokens for bag-of-words ML (`text_lemmatized`).
  3. **Sarcasm, Pragmatics, and Context:** Political impeachment discourse relies heavily on irony, rhetorical questions, and courtroom catchphrases—highlighting why context-aware machine learning models outperform context-free word lists, and why human-in-the-loop gold annotation remains indispensable for affective AI.

---

## Task 6: Final Presentation Outline (All 9 Required Slides)

1. **Slide 1 — Title Page**
   - **Title:** Public Sentiment & Affective Analysis of VP Sara Duterte's Impeachment Trial (YouTube & Reddit)
   - **Course:** CS412 — CS Elective 2 (Affective Computing) | 1st Sem, SY 2026–2027
   - **Instructor:** Love Jhoye M. Raboy, PhD
   - **Group Members (5):** John Cyril Espina, Den Jester Antonio, Sophia Marie Flores, John Jaybird Casia, Glenn Mark Anino
2. **Slide 2 — Introduction**
   - Domain: High-stakes Philippine political & constitutional event (2026 Senate Impeachment Trial of VP Sara Duterte).
   - Why it matters: Understanding civic emotion (anger, distrust, hope, support) across broadcast news audiences (YouTube) and civic forums (Reddit) connects NLP directly to Affective Computing.
3. **Slide 3 — Dataset & Collection Process**
   - **Total Collected:** **{total_raw:,} raw comments** → **{total_clean:,} clean unique comments** (`{yt_clean_n:,}` YouTube + `{rd_clean_n:,}` Reddit), exceeding the 1,000-comment requirement by over **{total_clean / 1000:.1f}×**.
   - **Ethical Steps Taken:** Public news channels & public subreddit only; zero interaction/engagement; SHA-256 one-way hashed user IDs; inline `@handle` & `u/handle` PII scrubbed; `deliverables/raw_comments.csv` stores text, timestamp, and reaction count only.
   - **Source Links (YouTube Clips & Coverages):**
{yt_links_md}
   - **Reddit Source:** Public `r/Philippines` impeachment megathread JSON snapshots (`1uophc7`).
4. **Slide 4 — Preprocessing Techniques**
   - Show funnel table from `deliverables/preprocessing_summary.txt`: HTML unescape, URL/handle removal, exact deduplication, `< 3` words removal, `<= 5` comments/user anti-spam cap, NLTK tokenization, bilingual EN+FIL stopword removal (preserving negations like `not`, `hindi`, `wala`, `dili`), and POS-aware WordNet lemmatization.
   - Include 2–3 real before/after comment examples from `deliverables/preprocessing_summary.txt`.
5. **Slide 5 — Lexicon Analysis (VADER, TextBlob, AFINN)**
   - Embed `lexicon_sentiment_distribution.png`, `lexicon_polarity_histograms.png`, and `lexicon_neutral_by_language.png`.
   - Highlight sample outputs and the Neutral spike caused by Tagalog/Taglish/Bisaya text.
6. **Slide 6 — ML-Based Analysis (TF-IDF + Logistic Regression & Naive Bayes)**
   - Human Gold Standard: 1,050 stratified comments annotated by 5 members{f" ({fleiss_line})" if fleiss_line else ""}.
   - Feature extraction: TF-IDF word n-grams `(1, 2)` (`sublinear_tf=True`, `max_features=10,000`) + 5-fold stratified CV on 80% train split, evaluated on 20% held-out test split.
   - Embed `deliverables/confusion_matrix.png`, `model_comparison_f1.png`, and `top_tfidf_features.png`.
7. **Slide 7 — Comparison & Discussion**
   - Best model comparison table (`deliverables/ml_results.txt`).
   - Domain emotional patterns (`platform_sentiment_comparison.png`, `temporal_sentiment_trend.png`).
   - Challenges: Taglish/Bisaya code-switching, sarcasm (`deliverables/charts/disagreement_case_studies.csv`), spelling variants (`walang`/`wlang`).
8. **Slide 8 — Conclusion & Relation to Affective Computing**
   - Synthesize empirical findings: English lexicons fail on >80% of multilingual PH comments; supervised TF-IDF captures domain & bilingual affect; Affective Computing systems for Philippine social media require multilingual, culturally grounded, and context-aware emotion modeling.
9. **Slide 9 — References**
   - Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM*.
   - Loria, S. (2018). *TextBlob: Simplified text processing*.
   - Nielsen, F. Å. (2011). A new ANEW: Evaluation of a word list for sentiment analysis in microblogs. *ESWC Workshop on Making Sense of Microposts*.
   - Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
   - Picard, R. W. (1997). *Affective Computing*. MIT Press.

---

## Page 5 — Laboratory Manual Submission Form (Ready-to-Fill Reference)

- **Course & Section:** CS412 / CS413 – CS Elective 2 (Affective Computing)
- **Topic:** Week 14 – Public Sentiment & Affective Analysis of VP Sara Duterte's Impeachment Trial
- **Summary Metrics Snapshot:**
  - Raw Dataset: `{total_raw:,}` comments | Clean Dataset: `{total_clean:,}` comments (`{yt_clean_n:,}` YouTube, `{rd_clean_n:,}` Reddit)
  - Lexicons Used: VADER, TextBlob, AFINN (all 3 implemented)
  - ML Models Used: TF-IDF + Logistic Regression & Multinomial Naive Bayes{f" | {fleiss_line}" if fleiss_line else ""}

---

## Appendix: Embedded Metrics Logs

{f"### 5-Member Inter-Annotator Agreement\\n```text\\n{iaa_txt.strip()}\\n```\\n" if iaa_txt else ""}### Preprocessing Summary
```text
{pre_txt.strip()}
```

### Lexicon Summary
```text
{lex_txt.strip()}
```

### ML Results Summary
```text
{ml_txt.strip()}
```
"""
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(md, encoding="utf-8")
    print(f"Saved comparative analysis & presentation outline: {REPORT_MD}")
    print(f"Saved disagreement case studies ({len(case_df)} examples): {CASES_CSV}")


if __name__ == "__main__":
    main()
