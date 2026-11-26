"""Stage 4: ML-Based Sentiment Analysis (TF-IDF + Logistic Regression & Multinomial Naive Bayes).

Supports:
  --mode auto         Uses data/annotated/gold_labels.csv if present; else provisional consensus
  --mode gold         Requires data/annotated/gold_labels.csv
  --mode provisional  Uses high-confidence lexicon consensus labels as a clearly labeled baseline
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

CLEAN_FILE = Path("data/processed/all_comments_clean.csv")
LEXICON_FILE = Path("deliverables/lexicon_results.csv")
GOLD_FILE = Path("data/annotated/gold_labels.csv")

RESULTS_TXT = Path("deliverables/ml_results.txt")
CONFUSION_PNG = Path("deliverables/confusion_matrix.png")
CHARTS_DIR = Path("deliverables/charts")
FULL_PRED_OUT = Path("data/processed/full_corpus_predictions.csv")

LABELS = ["Negative", "Neutral", "Positive"]
RANDOM_STATE = 42


def load_modeling_data(mode: str) -> tuple[pd.DataFrame, str]:
    """Load labeled modeling dataset along with lexicon predictions for baseline comparison."""
    clean_df = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    lex_df = pd.read_csv(LEXICON_FILE, encoding="utf-8-sig")
    if len(clean_df) != len(lex_df):
        raise RuntimeError("Row count mismatch between all_comments_clean.csv and lexicon_results.csv")

    merged = clean_df.copy()
    for col in [
        "vader_compound",
        "vader_label",
        "textblob_polarity",
        "textblob_label",
        "afinn_score",
        "afinn_label",
        "lexicon_majority_label",
        "lexicon_unanimous",
    ]:
        merged[col] = lex_df[col].values

    use_gold = mode == "gold" or (mode == "auto" and GOLD_FILE.exists())
    if use_gold:
        if not GOLD_FILE.exists():
            raise FileNotFoundError(f"Gold mode requested, but {GOLD_FILE} does not exist.")
        gold_df = pd.read_csv(GOLD_FILE, encoding="utf-8-sig")
        key_col = "comment_id" if "comment_id" in gold_df.columns and gold_df["comment_id"].notna().all() else "text_clean"
        merged[key_col] = merged[key_col].fillna("").astype(str)
        gold_df[key_col] = gold_df[key_col].fillna("").astype(str)
        sub = merged.merge(
            gold_df[[key_col, "gold_label"]].drop_duplicates(subset=[key_col]),
            on=key_col,
            how="inner",
        )
        sub["target_label"] = sub["gold_label"]
        label_source = "HUMAN GOLD STANDARD (5-Member Annotated)"
    else:
        # Provisional baseline: unanimous 3-way lexicon agreement first, else 2-of-3 non-Mixed
        unanimous = merged[merged["lexicon_unanimous"] & merged["lexicon_majority_label"].isin(LABELS)].copy()
        if len(unanimous) >= 1200:
            sub = unanimous
        else:
            sub = merged[merged["lexicon_majority_label"].isin(LABELS)].copy()
        sub["target_label"] = sub["lexicon_majority_label"]
        label_source = "PROVISIONAL BASELINE — PENDING 5-MEMBER HUMAN GOLD LABELS"

    sub["text_lemmatized"] = sub["text_lemmatized"].fillna("").astype(str).str.strip()
    sub = sub[sub["text_lemmatized"] != ""].copy()
    # Prevent train/test leakage from identical lemmatized strings
    sub = sub.drop_duplicates(subset=["text_lemmatized"], keep="first").reset_index(drop=True)
    return sub, label_source


def evaluate_predictions(name: str, y_true, y_pred) -> dict:
    return {
        "Method": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision (Macro)": precision_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "F1 (Weighted)": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
    }


def plot_confusion_matrices(y_test, pred_lr, pred_nb, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)
    models = [
        ("TF-IDF + Logistic Regression", pred_lr),
        ("TF-IDF + Multinomial Naive Bayes", pred_nb),
    ]
    for ax, (title, preds) in zip(axes, models):
        cm = confusion_matrix(y_test, preds, labels=LABELS)
        cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            cbar=False,
            ax=ax,
        )
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10.5)
        ax.set_ylabel("True Label", fontsize=10.5)

    fig.suptitle(f"Held-Out Test Confusion Matrices ({title_suffix})", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    CONFUSION_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comp_df: pd.DataFrame, title_suffix: str) -> None:
    melted = comp_df.melt(
        id_vars="Method",
        value_vars=["Accuracy", "F1 (Macro)", "F1 (Weighted)"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=300)
    sns.barplot(data=melted, x="Method", y="Score", hue="Metric", palette=["#264653", "#2A9D8F", "#E9C46A"], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    ax.set_title(f"Held-Out Test Performance: Lexicons vs. Traditional ML ({title_suffix})", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 – 1.0)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 1.12)
    plt.xticks(rotation=15, ha="right")
    sns.despine()
    fig.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / "model_comparison_f1.png", dpi=300)
    plt.close(fig)


def plot_top_tfidf_features(lr_pipe: Pipeline) -> list[str]:
    vec = lr_pipe.named_steps["tfidf"]
    clf = lr_pipe.named_steps["clf"]
    feature_names = np.array(vec.get_feature_names_out())
    classes = list(clf.classes_)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    lines = ["Top Logistic Regression TF-IDF Features by Class:"]

    for ax, target_cls, color in zip(axes, ["Negative", "Positive"], ["#D64550", "#2A9D8F"]):
        cls_idx = classes.index(target_cls)
        coefs = clf.coef_[cls_idx]
        top_idx = np.argsort(coefs)[-15:]
        top_words = feature_names[top_idx]
        top_vals = coefs[top_idx]

        ax.barh(top_words, top_vals, color=color, alpha=0.88)
        ax.set_title(f"Top 15 Predictive N-grams → {target_cls}", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Logistic Regression Weight", fontsize=10)
        sns.despine(ax=ax)

        lines.append(f"\n  {target_cls}: " + ", ".join(reversed(top_words.tolist())))

    # Also capture Neutral top words in text log
    if "Neutral" in classes:
        neu_idx = classes.index("Neutral")
        top_neu = feature_names[np.argsort(clf.coef_[neu_idx])[-15:]]
        lines.append("\n  Neutral:  " + ", ".join(reversed(top_neu.tolist())))

    fig.suptitle("Interpretable Affective N-grams Learned by TF-IDF + Logistic Regression", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "top_tfidf_features.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "gold", "provisional"], default="auto")
    args = parser.parse_args()

    df, label_source = load_modeling_data(args.mode)
    is_provisional = "PROVISIONAL" in label_source
    short_tag = "Provisional Consensus Baseline" if is_provisional else "5-Member Human Gold Standard"

    X = df["text_lemmatized"].values
    y = df["target_label"].values

    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    # Assert zero text leakage across splits
    overlap = set(X_train).intersection(set(X_test))
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping texts between train and test!"

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Logistic Regression Pipeline
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    lr_grid = GridSearchCV(
        lr_pipe,
        param_grid={"clf__C": [0.1, 0.5, 1.0, 2.0, 5.0]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    lr_cv_scores = cross_val_score(best_lr, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    pred_lr = best_lr.predict(X_test)

    # 2. Multinomial Naive Bayes Pipeline
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=10000)),
        ("clf", MultinomialNB()),
    ])
    nb_grid = GridSearchCV(
        nb_pipe,
        param_grid={"clf__alpha": [0.1, 0.25, 0.5, 1.0, 2.0]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    nb_grid.fit(X_train, y_train)
    best_nb = nb_grid.best_estimator_
    nb_cv_scores = cross_val_score(best_nb, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    pred_nb = best_nb.predict(X_test)

    # 3. Word + Character N-gram Ablation (Logistic Regression)
    union_pipe = Pipeline([
        (
            "features",
            FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.95, max_features=10000)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3, max_features=15000)),
            ]),
        ),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=lr_grid.best_params_["clf__C"], random_state=RANDOM_STATE)),
    ])
    union_pipe.fit(X_train, y_train)
    pred_union = union_pipe.predict(X_test)

    # Evaluate all 5 primary methods + ablation on the EXACT SAME held-out test set
    test_df = df.iloc[idx_test]
    metrics_rows = [
        evaluate_predictions("VADER (Lexicon)", y_test, test_df["vader_label"].values),
        evaluate_predictions("TextBlob (Lexicon)", y_test, test_df["textblob_label"].values),
        evaluate_predictions("AFINN (Lexicon)", y_test, test_df["afinn_label"].values),
        evaluate_predictions("Multinomial NB (TF-IDF)", y_test, pred_nb),
        evaluate_predictions("Logistic Regression (TF-IDF)", y_test, pred_lr),
        evaluate_predictions("LogReg (Word+Char TF-IDF Ablation)", y_test, pred_union),
    ]
    comp_df = pd.DataFrame(metrics_rows)

    # Charts
    plot_confusion_matrices(y_test, pred_lr, pred_nb, short_tag)
    plot_model_comparison(comp_df.iloc[:5], short_tag)
    feat_lines = plot_top_tfidf_features(best_lr)

    # Predict across full clean corpus using best primary rubric model (LR vs NB)
    lr_f1 = f1_score(y_test, pred_lr, labels=LABELS, average="macro", zero_division=0)
    nb_f1 = f1_score(y_test, pred_nb, labels=LABELS, average="macro", zero_division=0)
    winner_name = "Logistic Regression (TF-IDF)" if lr_f1 >= nb_f1 else "Multinomial Naive Bayes (TF-IDF)"
    winner_pipe = best_lr if lr_f1 >= nb_f1 else best_nb

    # Refit winning pipeline on all labeled data for full-corpus affective inference
    full_clean = pd.read_csv(CLEAN_FILE, encoding="utf-8-sig")
    lex_all = pd.read_csv(LEXICON_FILE, encoding="utf-8-sig")
    winner_pipe.fit(X, y)

    full_pred_df = full_clean.copy()
    for col in ["vader_compound", "vader_label", "textblob_polarity", "textblob_label", "afinn_score", "afinn_label", "lexicon_majority_label"]:
        full_pred_df[col] = lex_all[col].values

    full_texts = full_pred_df["text_lemmatized"].fillna("").astype(str)
    full_pred_df["ml_predicted_label"] = winner_pipe.predict(full_texts)
    probs = winner_pipe.predict_proba(full_texts)
    for cls_i, cls_name in enumerate(winner_pipe.classes_):
        full_pred_df[f"ml_prob_{cls_name.lower()}"] = np.round(probs[:, cls_i], 4)
    full_pred_df["ml_model_used"] = winner_name
    full_pred_df["ml_label_source"] = label_source

    FULL_PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    full_pred_df.to_csv(FULL_PRED_OUT, index=False, encoding="utf-8-sig")

    # Format deliverables/ml_results.txt
    lines = []
    lines.append("=" * 78)
    lines.append("CS412 WEEK 14 PIT — TASK 4: ML-BASED SENTIMENT ANALYSIS RESULTS")
    lines.append(f"Label Source: {label_source}")
    if is_provisional:
        lines.append("NOTE: Run `python src/04_modeling/merge_annotations.py` once the 5-member")
        lines.append("      annotation sheets are filled, then re-run `train_ml_models.py`.")
    lines.append("=" * 78)
    lines.append(f"\nDataset Size (deduplicated lemmatized): {len(df):,}")
    lines.append(f"Train Split (80%): {len(X_train):,} | Held-Out Test Split (20%): {len(X_test):,}")
    lines.append(f"Train/Test Text Overlap: {len(overlap)} (verified zero leakage)")
    lines.append("\nClass Distribution in Modeling Dataset:")
    for lbl in LABELS:
        c = int((y == lbl).sum())
        lines.append(f"  {lbl:<10}: {c:>5,} ({c / len(y) * 100:.1f}%)")

    lines.append("\n--- 5-Fold Stratified Cross-Validation (Train Split Only) ---")
    lines.append(f"Logistic Regression best params: {lr_grid.best_params_} | CV Macro-F1: {lr_cv_scores.mean():.4f} +/- {lr_cv_scores.std():.4f}")
    lines.append(f"Multinomial NB best params:      {nb_grid.best_params_} | CV Macro-F1: {nb_cv_scores.mean():.4f} +/- {nb_cv_scores.std():.4f}")

    lines.append("\n--- Held-Out Test Set Comparison (All Methods on Identical 20% Test Split) ---")
    comp_fmt = comp_df.copy()
    for c in ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)", "F1 (Weighted)"]:
        comp_fmt[c] = comp_fmt[c].map(lambda v: f"{v:.4f}")
    lines.append(comp_fmt.to_string(index=False))

    lines.append(f"\nBest Primary ML Model: {winner_name}")

    lines.append("\n--- Detailed Classification Report: TF-IDF + Logistic Regression ---")
    lines.append(classification_report(y_test, pred_lr, labels=LABELS, digits=4, zero_division=0))
    lines.append("Confusion Matrix (rows=True [Neg, Neu, Pos], cols=Pred [Neg, Neu, Pos]):")
    lines.append(str(confusion_matrix(y_test, pred_lr, labels=LABELS)))

    lines.append("\n--- Detailed Classification Report: TF-IDF + Multinomial Naive Bayes ---")
    lines.append(classification_report(y_test, pred_nb, labels=LABELS, digits=4, zero_division=0))
    lines.append("Confusion Matrix (rows=True [Neg, Neu, Pos], cols=Pred [Neg, Neu, Pos]):")
    lines.append(str(confusion_matrix(y_test, pred_nb, labels=LABELS)))

    lines.append("\n--- Interpretability ---")
    lines.extend(feat_lines)

    lines.append("\n--- Full-Corpus ML Sentiment Distribution (n = {:,}) ---".format(len(full_pred_df)))
    for lbl in LABELS:
        c = int((full_pred_df["ml_predicted_label"] == lbl).sum())
        lines.append(f"  {lbl:<10}: {c:>6,} ({c / len(full_pred_df) * 100:.1f}%)")

    report_text = "\n".join(lines) + "\n"
    RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_TXT.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Saved ML results:          {RESULTS_TXT}")
    print(f"Saved confusion matrix:    {CONFUSION_PNG}")
    print(f"Saved full predictions:    {FULL_PRED_OUT}")


if __name__ == "__main__":
    main()
