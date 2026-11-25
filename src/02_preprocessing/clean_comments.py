"""Unified Stage 2 NLP preprocessing pipeline for multi-platform impeachment comments.

Performs:
1. Cleaning (HTML unescape, NFKC normalization, URL/mention/timestamp stripping, elongation normalization)
2. Quality & anti-spam filtering (empty removal, deduplication, >= 3 words, <= 5 comments/user)
3. Tokenization (NLTK word_tokenize)
4. Bilingual EN + FIL/Bisaya stopword removal (strictly preserving negations & intensifiers)
5. POS-aware WordNet lemmatization
6. Affective metadata & language-bucket tagging
"""

import html
import re
import unicodedata
from pathlib import Path

import emoji
import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

INPUT_FILE = Path("data/raw/all_comments_raw.csv")
ALL_CLEAN_OUT = Path("data/processed/all_comments_clean.csv")
YT_CLEAN_OUT = Path("data/processed/youtube_comments_clean.csv")
DELIVERABLE_CLEAN_OUT = Path("deliverables/cleaned_comments.csv")
SUMMARY_OUT = Path("deliverables/preprocessing_summary.txt")

MIN_WORDS = 3
MAX_COMMENTS_PER_USER = 5

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
AT_MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_.\-]+")
REDDIT_USER_PATTERN = re.compile(r"(?:^|(?<=\s))/?u/[A-Za-z0-9_\-]+", re.IGNORECASE)
TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
REPEAT_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
NON_WORD_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)

# Functional Filipino / Tagalog / Bisaya stopwords (excluding negations & intensifiers)
FILIPINO_STOPWORDS = {
    "ang", "ng", "mga", "sa", "na", "ay", "ko", "mo", "niya", "sila", "tayo",
    "kami", "kayo", "ito", "iyan", "yun", "yung", "ung", "lang", "naman", "pero",
    "kung", "dahil", "para", "po", "ba", "eh", "kasi", "pa", "rin", "din", "ni",
    "kay", "si", "sina", "pag", "kapag", "yan", "dito", "dyan", "doon", "siya",
    "nya", "natin", "ako", "ikaw", "ka", "tapos", "habang", "kahit", "nang",
    "ug", "og", "nga", "man", "ra", "gud", "bitaw", "lagi", "mao", "kani",
    "kana", "kini", "at", "o", "may", "meron", "nila", "namin", "amin", "atin",
    "akin", "iyo", "kanila", "kanya", "kanyang", "aking", "iyong", "ating",
    "aming", "mga", "ung", "yung", "yan", "yun", "eto", "heto", "ayan", "ayun",
    "daw", "raw", "pala", "kaya", "sana", "basta", "mukhang", "parang", "tuloy",
    "muna", "talaga", "naging", "maging", "mag", "nag", "um", "in",
}

NEGATIONS_AND_INTENSIFIERS = {
    "not", "no", "nor", "never", "none", "nothing", "neither", "cannot",
    "cant", "can't", "dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
    "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't", "werent", "weren't",
    "wont", "won't", "wouldnt", "wouldn't", "shouldnt", "shouldn't",
    "couldn't", "couldnt", "ain't", "aint",
    "hindi", "di", "wala", "walang", "dili", "huwag", "wag", "mas", "sobra", "grabe",
}

TAGALOG_MARKERS = re.compile(
    r"\b(ang|ng|mga|sa|na|ni|si|ay|ko|mo|ka|po|lang|naman|din|rin|talaga|pero|"
    r"kasi|para|yung|yan|ito|dito|kung|hindi|di|wala|meron|may|sila|tayo|kami|"
    r"kayo|nila|natin|bakit|paano|sino|ano|kelan|dapat|huwag|wag|grabe|sana|"
    r"tuloy|bayan|gobyerno|pilipinas|pinas|puro|nang|dahil|kaya|masyado|sobra|"
    r"nga|ba|eh|daw|raw|palagi|kahit|habang|kapag|pag|basta|mukhang|alam|"
    r"sabihin|sinabi|gawin|ginawa|tao|taong|pera|bilyon|milyon|nakaw|"
    r"magnanakaw|korap|kurakot|kaso|senado|kongreso|pangulo|presidente|bise)\b",
    re.IGNORECASE,
)

BISAYA_MARKERS = re.compile(
    r"\b(kaayo|gyud|jud|man|gud|unsa|ngano|diri|didto|kini|kana|kamo|naa|"
    r"dili|diay|bitaw|lagi|baya|ug|og|ra|mao|kani|adto|sige|bisaya|cebu|davao)\b",
    re.IGNORECASE,
)

ENGLISH_CONTENT_MARKERS = re.compile(
    r"\b(the|is|are|was|were|be|been|have|has|had|will|would|should|could|"
    r"that|this|these|those|with|from|they|their|them|about|because|what|"
    r"where|when|which|who|why|how|not|for|and|but|you|your|she|her|he|his|"
    r"trial|impeachment|senate|defense|prosecution|evidence|court|money|guilty|"
    r"corrupt|people|country|taxpayers|witness|truth|justice|law|vote)\b",
    re.IGNORECASE,
)


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if missing."""
    required = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
        "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    }
    for res_path, pkg in required.items():
        try:
            nltk.data.find(res_path)
        except LookupError:
            nltk.download(pkg, quiet=True)


def build_stopword_set() -> set[str]:
    en_stops = set(stopwords.words("english"))
    # Strip apostrophe variants as well for normalized tokens
    en_no_apos = {w.replace("'", "") for w in en_stops}
    combined = en_stops | en_no_apos | FILIPINO_STOPWORDS
    return combined - NEGATIONS_AND_INTENSIFIERS


def to_wordnet_pos(treebank_tag: str) -> str:
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_preserve_affect(text: str) -> str:
    """Light cleaning: unescape HTML, strip URLs/@/u/handles, keep emojis, caps, !/?."""
    s = html.unescape(str(text or ""))
    s = unicodedata.normalize("NFKC", s)
    s = URL_PATTERN.sub(" ", s)
    s = AT_MENTION_PATTERN.sub(" ", s)
    s = REDDIT_USER_PATTERN.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_nopunct(text_clean: str) -> str:
    """Lowercase normalized text without punctuation/digits for tokenization."""
    s = text_clean.lower()
    s = TIMESTAMP_PATTERN.sub(" ", s)
    s = NON_WORD_PATTERN.sub(" ", s)
    s = re.sub(r"\d+", " ", s)
    s = REPEAT_CHAR_PATTERN.sub(r"\1\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_language_bucket(text_clean: str) -> str:
    tl_hits = len(TAGALOG_MARKERS.findall(text_clean))
    bis_hits = len(BISAYA_MARKERS.findall(text_clean))
    en_hits = len(ENGLISH_CONTENT_MARKERS.findall(text_clean))

    if bis_hits >= 2 or (bis_hits >= 1 and tl_hits == 0 and en_hits <= 1):
        return "Bisaya/Other"
    if tl_hits >= 2 and en_hits >= 2:
        return "Taglish/Code-switched"
    if tl_hits >= 2:
        return "Filipino/Tagalog"
    if tl_hits == 1 and en_hits >= 2:
        return "Taglish/Code-switched"
    if tl_hits == 1 and en_hits <= 1:
        return "Filipino/Tagalog"
    if en_hits >= 1:
        return "English"
    return "Bisaya/Other"


def process_nlp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply tokenization, bilingual stopword removal, and POS-aware lemmatization."""
    stop_set = build_stopword_set()
    lemmatizer = WordNetLemmatizer()

    tokens_col = []
    token_counts = []
    tokens_no_stop_col = []
    lemmatized_col = []

    for norm_text in df["text_cleaned_nopunct"]:
        raw_toks = [t for t in word_tokenize(norm_text) if any(ch.isalpha() for ch in t)]
        filtered_toks = [
            t
            for t in raw_toks
            if (t not in stop_set and len(t) > 1) or (t in NEGATIONS_AND_INTENSIFIERS)
        ]

        # Ensure non-empty fallback if stopword filter removed everything
        if not filtered_toks:
            filtered_toks = raw_toks

        tagged = pos_tag(filtered_toks)
        lemmas = [lemmatizer.lemmatize(tok, to_wordnet_pos(tag)) for tok, tag in tagged]

        tokens_col.append(" ".join(raw_toks))
        token_counts.append(len(raw_toks))
        tokens_no_stop_col.append(" ".join(filtered_toks))
        lemmatized_col.append(" ".join(lemmas))

    df["tokens"] = tokens_col
    df["token_count"] = token_counts
    df["tokens_no_stop"] = tokens_no_stop_col
    df["text_lemmatized"] = lemmatized_col
    return df


def main():
    ensure_nltk_resources()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run Stage 1 merge first.")

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    funnel = []

    def record_funnel(label: str, frame: pd.DataFrame):
        counts = frame["platform"].value_counts().to_dict() if not frame.empty else {}
        funnel.append((label, len(frame), counts))

    record_funnel("0. Raw merged input", df)

    # 1. Cleaning & empty/URL-only removal
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].map(clean_preserve_affect)
    df["text_cleaned_nopunct"] = df["text_clean"].map(clean_nopunct)
    df = df[df["text_cleaned_nopunct"] != ""].copy()
    record_funnel("1. After cleaning & removing empty/URL-only", df)

    # 2. Sort by likes desc, timestamp asc, then drop exact normalized duplicates
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
    df["timestamp_utc"] = df["timestamp_utc"].fillna("").astype(str)
    df = df.sort_values(by=["likes", "timestamp_utc"], ascending=[False, True])
    df = df.drop_duplicates(subset=["text_cleaned_nopunct"], keep="first").copy()
    record_funnel("2. After exact normalized-text deduplication", df)

    # 3. Minimum word count filter (>= MIN_WORDS)
    df["word_count"] = df["text_cleaned_nopunct"].str.split().str.len()
    df = df[df["word_count"] >= MIN_WORDS].copy()
    record_funnel(f"3. After keeping >= {MIN_WORDS} words", df)

    # 4. Cap comments per unique (platform, user_hash)
    df["user_hash"] = df["user_hash"].fillna("unknown").astype(str)
    is_unknown = df["user_hash"].isin(["unknown", "anon_unknown", ""])
    known_users = (
        df[~is_unknown]
        .groupby(["platform", "user_hash"], group_keys=False)
        .head(MAX_COMMENTS_PER_USER)
    )
    df = pd.concat([known_users, df[is_unknown]], ignore_index=True)
    df = df.sort_values(by=["platform", "likes"], ascending=[True, False]).reset_index(drop=True)
    record_funnel(f"4. After capping <= {MAX_COMMENTS_PER_USER} comments/user", df)

    # 5. Tokenization, bilingual stopword removal, POS-aware lemmatization
    df = process_nlp_columns(df)
    df = df[df["text_lemmatized"].str.strip() != ""].copy().reset_index(drop=True)
    record_funnel("5. Final NLP-preprocessed dataset", df)

    # 6. Affective & language metadata
    df["char_count"] = df["text_clean"].str.len()
    df["exclamation_count"] = df["text_clean"].str.count("!")
    df["question_count"] = df["text_clean"].str.count(r"\?")
    df["upper_ratio"] = df["text_clean"].map(
        lambda s: round(sum(c.isupper() for c in s) / max(sum(c.isalpha() for c in s), 1), 3)
    )
    df["emoji_count"] = df["text_clean"].map(emoji.emoji_count)
    df["language_bucket"] = df["text_clean"].map(detect_language_bucket)

    # Save outputs
    ALL_CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALL_CLEAN_OUT, index=False, encoding="utf-8-sig")

    yt_df = df[df["platform"] == "youtube"].copy()
    yt_df.to_csv(YT_CLEAN_OUT, index=False, encoding="utf-8-sig")

    deliverable_cols = [
        "platform",
        "timestamp_utc",
        "likes",
        "text_clean",
        "tokens",
        "tokens_no_stop",
        "text_lemmatized",
        "word_count",
        "language_bucket",
    ]
    DELIVERABLE_CLEAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    df[deliverable_cols].to_csv(DELIVERABLE_CLEAN_OUT, index=False, encoding="utf-8-sig")

    # Write human-readable preprocessing summary
    lines = ["=== Stage 2: Unified NLP Preprocessing Summary ===\n"]
    lines.append(f"{'Stage':<48} | {'Total':>7} | {'YouTube':>8} | {'Reddit':>7}")
    lines.append("-" * 79)
    for stage, total, counts in funnel:
        yt_c = counts.get("youtube", 0)
        rd_c = counts.get("reddit", 0)
        lines.append(f"{stage:<48} | {total:>7,} | {yt_c:>8,} | {rd_c:>7,}")

    lines.append("\n--- Clean Dataset by Platform ---")
    lines.append(df["platform"].value_counts().to_string())
    lines.append("\n--- Clean Dataset by Language Bucket ---")
    lang_vc = df["language_bucket"].value_counts()
    for lang, cnt in lang_vc.items():
        lines.append(f"  {lang:<25}: {cnt:>6,} ({cnt / len(df) * 100:.1f}%)")

    lines.append(f"\nMedian words/comment: {df['word_count'].median():.1f}")
    lines.append(f"Mean words/comment:   {df['word_count'].mean():.1f}")
    lines.append(f"Unique user hashes:   {df['user_hash'].nunique():,}")

    lines.append("\n--- Sample Before / After NLP Transformations (for Task 6 Slides) ---")
    samples = df.sample(n=min(6, len(df)), random_state=42)
    for idx, (_, row) in enumerate(samples.iterrows(), start=1):
        lines.append(f"\nExample {idx} [{row['platform']} | {row['language_bucket']}]:")
        lines.append(f"  text_clean:      {row['text_clean'][:140]}")
        lines.append(f"  tokens:          {row['tokens'][:140]}")
        lines.append(f"  tokens_no_stop:  {row['tokens_no_stop'][:140]}")
        lines.append(f"  text_lemmatized: {row['text_lemmatized'][:140]}")

    summary_text = "\n".join(lines) + "\n"
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Saved full clean CSV:     {ALL_CLEAN_OUT}")
    print(f"Saved YouTube clean CSV:  {YT_CLEAN_OUT}")
    print(f"Saved rubric deliverable: {DELIVERABLE_CLEAN_OUT}")
    print(f"Saved summary report:     {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
