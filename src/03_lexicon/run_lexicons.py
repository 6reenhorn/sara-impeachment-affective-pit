"""Stage 3: Lexicon-Based Sentiment Analysis (VADER, TextBlob, AFINN).

Computes polarity scores and Positive/Negative/Neutral labels on lightly-cleaned
affect-preserving text (`text_clean`), logs inter-lexicon agreement and Neutral
inflation by language bucket, and exports deliverables/lexicon_results.csv + charts.
"""

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from afinn import Afinn
from sklearn.metrics import cohen_kappa_score
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE = Path("data/processed/all_comments_clean.csv")
OUTPUT_CSV = Path("deliverables/lexicon_results.csv")
SUMMARY_TXT = Path("deliverables/lexicon_summary.txt")
CHARTS_DIR = Path("deliverables/charts")

LABELS = ["Negative", "Neutral", "Positive"]
PALETTE = {"Negative": "#D64550", "Neutral": "#8D99AE", "Positive": "#2A9D8F"}


def vader_classify(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def textblob_classify(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


def afinn_classify(score: float) -> str:
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def majority_vote(row) -> tuple[str, bool]:
    votes = [row["vader_label"], row["textblob_label"], row["afinn_label"]]
    counts = Counter(votes)
    top_label, top_count = counts.most_common(1)[0]
    unanimous = top_count == 3
    if top_count >= 2:
        return top_label, unanimous
    return "Mixed", False


def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    lexicons = [
        ("VADER", "vader_label"),
        ("TextBlob", "textblob_label"),
        ("AFINN", "afinn_label"),
    ]
    rows = []
    for name, col in lexicons:
        vc = df[col].value_counts(normalize=True) * 100
        for lbl in LABELS:
            rows.append({"Lexicon": name, "Sentiment": lbl, "Percentage": vc.get(lbl, 0.0)})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=300)
    sns.barplot(
        data=plot_df,
        x="Lexicon",
        y="Percentage",
        hue="Sentiment",
        hue_order=LABELS,
        palette=PALETTE,
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=8.5)
    ax.set_title(
        f"Lexicon Sentiment Distribution Across Corpus (n = {len(df):,})",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Share of Comments (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, min(100, plot_df["Percentage"].max() + 12))
    ax.legend(title="Sentiment", frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lexicon_sentiment_distribution.png", dpi=300)
    plt.close(fig)


def plot_polarity_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=300)

    specs = [
        ("VADER Compound Score", df["vader_compound"], (-1.0, 1.0), 0.05, -0.05, "#264653"),
        ("TextBlob Polarity", df["textblob_polarity"], (-1.0, 1.0), 0.05, -0.05, "#457B9D"),
        (
            "AFINN Normalized Polarity",
            df["afinn_polarity"].clip(-1.5, 1.5),
            (-1.5, 1.5),
            0.0,
            0.0,
            "#E76F51",
        ),
    ]
    for ax, (title, series, xlim, pos_th, neg_th, color) in zip(axes, specs):
        ax.hist(series, bins=35, color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(pos_th, color="#2A9D8F", linestyle="--", linewidth=1.2)
        if neg_th != pos_th:
            ax.axvline(neg_th, color="#D64550", linestyle="--", linewidth=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylabel("Comment Count", fontsize=10)
        ax.set_xlim(xlim)
        sns.despine(ax=ax)

    fig.suptitle(
        "Polarity Score Distributions Across Lexicons (Zero-Spike Reflects Tagalog/Taglish OOV)",
        fontsize=12.5,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lexicon_polarity_histograms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_agreement_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [("VADER", "vader_label"), ("TextBlob", "textblob_label"), ("AFINN", "afinn_label")]
    names = [n for n, _ in cols]
    pct_mat = pd.DataFrame(np.ones((3, 3)) * 100.0, index=names, columns=names)
    kap_mat = pd.DataFrame(np.ones((3, 3)), index=names, columns=names)

    annot = pd.DataFrame("", index=names, columns=names)
    for i, (n1, c1) in enumerate(cols):
        for j, (n2, c2) in enumerate(cols):
            pct = (df[c1] == df[c2]).mean() * 100
            kap = cohen_kappa_score(df[c1], df[c2]) if i != j else 1.0
            pct_mat.loc[n1, n2] = pct
            kap_mat.loc[n1, n2] = kap
            annot.loc[n1, n2] = f"{pct:.1f}%\n(κ={kap:.2f})"

    fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=300)
    sns.heatmap(
        kap_mat,
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Cohen's Kappa (κ)"},
        ax=ax,
    )
    ax.set_title("Inter-Lexicon Agreement (% Match & Cohen's κ)", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lexicon_agreement_matrix.png", dpi=300)
    plt.close(fig)
    return pct_mat, kap_mat


def plot_neutral_by_language(df: pd.DataFrame) -> pd.DataFrame:
    order = ["English", "Taglish/Code-switched", "Filipino/Tagalog", "Bisaya/Other"]
    present_order = [o for o in order if o in df["language_bucket"].unique()]

    rows = []
    for lang in present_order:
        sub = df[df["language_bucket"] == lang]
        for name, col in [("VADER", "vader_label"), ("TextBlob", "textblob_label"), ("AFINN", "afinn_label")]:
            neu_rate = (sub[col] == "Neutral").mean() * 100
            rows.append({"Language": lang, "Lexicon": name, "Neutral Rate (%)": neu_rate, "n": len(sub)})

    neu_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    sns.barplot(
        data=neu_df,
        x="Language",
        y="Neutral Rate (%)",
        hue="Lexicon",
        palette=["#264653", "#457B9D", "#E76F51"],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=2, fontsize=8)
    ax.set_title(
        "English-Lexicon Blindness: Neutral Rate by Language Bucket",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Comments Classified as Neutral (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    sns.despine()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "lexicon_neutral_by_language.png", dpi=300)
    plt.close(fig)
    return neu_df


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run Stage 2 preprocessing first.")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df["text_lemmatized"] = df["text_lemmatized"].fillna("").astype(str)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(1).clip(lower=1)

    vader = SentimentIntensityAnalyzer()
    afinn = Afinn(emoticons=True)

    # 1. VADER on affect-preserving text_clean
    vader_scores = df["text_clean"].map(vader.polarity_scores)
    df["vader_pos"] = vader_scores.map(lambda d: round(d["pos"], 4))
    df["vader_neu"] = vader_scores.map(lambda d: round(d["neu"], 4))
    df["vader_neg"] = vader_scores.map(lambda d: round(d["neg"], 4))
    df["vader_compound"] = vader_scores.map(lambda d: round(d["compound"], 4))
    df["vader_label"] = df["vader_compound"].map(vader_classify)

    # 2. TextBlob on text_clean (+ ablation check vs text_lemmatized)
    tb_res = df["text_clean"].map(lambda s: TextBlob(s).sentiment)
    df["textblob_polarity"] = tb_res.map(lambda s: round(float(s.polarity), 4))
    df["textblob_subjectivity"] = tb_res.map(lambda s: round(float(s.subjectivity), 4))
    df["textblob_label"] = df["textblob_polarity"].map(textblob_classify)

    tb_lem_labels = df["text_lemmatized"].map(lambda s: textblob_classify(float(TextBlob(s).sentiment.polarity)))
    tb_clean_vs_lem_match = (df["textblob_label"] == tb_lem_labels).mean() * 100

    # 3. AFINN on text_clean
    df["afinn_score"] = df["text_clean"].map(lambda s: float(afinn.score(s)))
    df["afinn_polarity"] = (df["afinn_score"] / df["word_count"]).round(4)
    df["afinn_label"] = df["afinn_score"].map(afinn_classify)

    # 4. Consensus fields
    votes = df.apply(majority_vote, axis=1)
    df["lexicon_majority_label"] = votes.map(lambda x: x[0])
    df["lexicon_unanimous"] = votes.map(lambda x: x[1])

    # Export deliverables/lexicon_results.csv
    export_cols = [
        "platform",
        "source_name",
        "topic_tag",
        "timestamp_utc",
        "likes",
        "word_count",
        "language_bucket",
        "text_clean",
        "vader_pos",
        "vader_neu",
        "vader_neg",
        "vader_compound",
        "vader_label",
        "textblob_polarity",
        "textblob_subjectivity",
        "textblob_label",
        "afinn_score",
        "afinn_polarity",
        "afinn_label",
        "lexicon_majority_label",
        "lexicon_unanimous",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df[export_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Charts & diagnostics
    plot_sentiment_distribution(df)
    plot_polarity_histograms(df)
    pct_mat, kap_mat = plot_agreement_matrix(df)
    neu_df = plot_neutral_by_language(df)

    # Write summary text
    lines = [f"=== Stage 3: Lexicon-Based Sentiment Summary (n = {len(df):,}) ===\n"]
    for name, col in [("VADER", "vader_label"), ("TextBlob", "textblob_label"), ("AFINN", "afinn_label")]:
        vc = df[col].value_counts()
        lines.append(f"{name}:")
        for lbl in LABELS:
            c = vc.get(lbl, 0)
            lines.append(f"  {lbl:<10}: {c:>6,} ({c / len(df) * 100:.1f}%)")

    lines.append(f"\nUnanimous 3-way agreement: {df['lexicon_unanimous'].sum():,} ({df['lexicon_unanimous'].mean()*100:.1f}%)")
    lines.append(f"TextBlob clean vs lemmatized label match: {tb_clean_vs_lem_match:.1f}%")
    lines.append("\nPairwise % Agreement:")
    lines.append(pct_mat.round(1).to_string())
    lines.append("\nPairwise Cohen's Kappa:")
    lines.append(kap_mat.round(3).to_string())
    lines.append("\nNeutral Rate by Language Bucket:")
    lines.append(neu_df.pivot(index="Language", columns="Lexicon", values="Neutral Rate (%)").round(1).to_string())

    summary_txt = "\n".join(lines) + "\n"
    SUMMARY_TXT.write_text(summary_txt, encoding="utf-8")
    print(summary_txt)
    print(f"Saved lexicon results: {OUTPUT_CSV}")
    print(f"Saved charts to:       {CHARTS_DIR}")


if __name__ == "__main__":
    main()
