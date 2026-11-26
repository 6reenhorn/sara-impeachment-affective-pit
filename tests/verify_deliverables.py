"""Automated verification suite for CS412 Week 14 PIT deliverables."""

import re
from pathlib import Path

import pandas as pd

DELIVERABLES = Path("deliverables")
CHARTS_DIR = DELIVERABLES / "charts"
SHEETS_DIR = Path("data/annotated/sheets")

VALID_LABELS = {"Positive", "Negative", "Neutral"}
ALLOWED_RAW_COLS = ["platform", "timestamp_utc", "likes", "text"]
FORBIDDEN_PII_COLS = {"author", "username", "user_id", "channel_id", "profile_url", "author_hash", "user_hash"}


def test_required_deliverables_exist():
    required_files = [
        DELIVERABLES / "raw_comments.csv",
        DELIVERABLES / "cleaned_comments.csv",
        DELIVERABLES / "lexicon_results.csv",
        DELIVERABLES / "ml_results.txt",
        DELIVERABLES / "confusion_matrix.png",
        DELIVERABLES / "comparative_analysis.md",
        CHARTS_DIR / "lexicon_sentiment_distribution.png",
        CHARTS_DIR / "lexicon_polarity_histograms.png",
        CHARTS_DIR / "lexicon_agreement_matrix.png",
        CHARTS_DIR / "lexicon_neutral_by_language.png",
        CHARTS_DIR / "model_comparison_f1.png",
        CHARTS_DIR / "top_tfidf_features.png",
        CHARTS_DIR / "platform_sentiment_comparison.png",
        CHARTS_DIR / "temporal_sentiment_trend.png",
        CHARTS_DIR / "affective_cues_by_sentiment.png",
        CHARTS_DIR / "disagreement_case_studies.csv",
    ]
    for path in required_files:
        assert path.exists(), f"Missing required deliverable: {path}"
        assert path.stat().st_size > 0, f"Empty deliverable: {path}"


def test_raw_deliverable_ethics_and_volume():
    raw = pd.read_csv(DELIVERABLES / "raw_comments.csv", encoding="utf-8-sig")
    assert list(raw.columns) == ALLOWED_RAW_COLS, f"Unexpected raw_comments.csv cols: {list(raw.columns)}"
    assert FORBIDDEN_PII_COLS.isdisjoint(set(raw.columns)), "PII column leaked into raw_comments.csv"
    assert len(raw) >= 1000, f"Expected >= 1000 raw comments, got {len(raw)}"
    assert raw["text"].isna().sum() == 0, "Null text in raw_comments.csv"

    texts = raw["text"].astype(str)
    at_leaks = texts.str.contains(r"@[A-Za-z0-9_.\-]+", regex=True).sum()
    u_leaks = texts.str.contains(r"(?:^|\s)/?u/[A-Za-z0-9_\-]+", regex=True).sum()
    assert at_leaks == 0, f"Found {at_leaks} @handle leaks in raw_comments.csv"
    assert u_leaks == 0, f"Found {u_leaks} u/handle leaks in raw_comments.csv"


def test_cleaned_deliverable_and_nlp_steps():
    clean = pd.read_csv(DELIVERABLES / "cleaned_comments.csv", encoding="utf-8-sig")
    assert len(clean) >= 8000, f"Expected >= 8000 expanded clean comments, got {len(clean)}"
    for col in ["text_clean", "tokens", "tokens_no_stop", "text_lemmatized"]:
        assert col in clean.columns, f"Missing column {col} in cleaned_comments.csv"
        assert clean[col].isna().sum() == 0, f"Nulls in {col}"

    texts = clean["text_clean"].astype(str)
    assert not texts.str.contains(r"https?://|www\.", regex=True).any(), "URL found in text_clean"
    assert not texts.str.contains(r"&(?:amp|gt|lt|quot);", regex=True).any(), "Unescaped HTML entity in text_clean"

    # Negations must survive stopword removal + lemmatization
    all_lemmas = " ".join(clean["text_lemmatized"].astype(str).tolist())
    for neg in ["not", "hindi", "wala"]:
        assert re.search(rf"\b{neg}\b", all_lemmas), f"Negation token '{neg}' missing from lemmatized corpus"


def test_lexicon_results_validity():
    lex = pd.read_csv(DELIVERABLES / "lexicon_results.csv", encoding="utf-8-sig")
    assert len(lex) >= 8000
    assert lex["vader_compound"].between(-1.0, 1.0).all(), "VADER compound out of [-1, 1]"
    assert lex["textblob_polarity"].between(-1.0, 1.0).all(), "TextBlob polarity out of [-1, 1]"
    for col in ["vader_label", "textblob_label", "afinn_label"]:
        assert set(lex[col].unique()).issubset(VALID_LABELS), f"Invalid label in {col}"


def test_ml_results_and_no_leakage():
    txt = (DELIVERABLES / "ml_results.txt").read_text(encoding="utf-8")
    for required in [
        "Logistic Regression",
        "Multinomial Naive Bayes",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Confusion Matrix",
        "Train/Test Text Overlap: 0",
    ]:
        assert required in txt, f"Missing '{required}' in deliverables/ml_results.txt"


def test_five_member_annotation_sheets():
    sheets = [pd.read_csv(SHEETS_DIR / f"member_{i}.csv", encoding="utf-8-sig") for i in range(1, 6)]
    for i, df in enumerate(sheets, start=1):
        assert len(df) == 250, f"member_{i}.csv has {len(df)} rows instead of 250"

    shared_sets = [set(df[df["is_shared_calibration"] == True]["sample_id"]) for df in sheets]
    unique_sets = [set(df[df["is_shared_calibration"] == False]["sample_id"]) for df in sheets]

    assert all(len(s) == 50 for s in shared_sets), "Each sheet must have 50 shared calibration rows"
    assert all(s == shared_sets[0] for s in shared_sets), "Shared calibration sample_ids must match across all 5 members"
    assert all(len(u) == 200 for u in unique_sets), "Each sheet must have 200 unique rows"

    union_unique = set().union(*unique_sets)
    assert len(union_unique) == 1000, f"Unique rows must be disjoint (expected 1000, got {len(union_unique)})"
    assert union_unique.isdisjoint(shared_sets[0]), "Shared calibration IDs overlapped with unique IDs"
