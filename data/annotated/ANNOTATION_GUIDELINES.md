# CS412 PIT — 5-Member Human Gold Standard Annotation Guidelines

Thank you for annotating! Each member has **250 comments** (`member_1.csv` .. `member_5.csv`):
- **50 shared calibration comments** (`is_shared_calibration = True`) appear in all 5 files so we can compute **Inter-Annotator Agreement (Fleiss' κ / Cohen's κ)**.
- **200 unique comments** belong only to your file (giving us **1,050 unique gold-labeled comments** total!).

## How to Fill Your CSV
Open your assigned CSV in Excel / Google Sheets (UTF-8) and fill the **`human_label`** column with exactly one of:
- `Positive`
- `Negative`
- `Neutral`

Optional: use the `notes` column to flag **sarcasm**, **Taglish/Bisaya idioms**, or **mixed feelings** (these make great examples for Task 5 & Presentation slides!).

---

## Labeling Rules (Sentiment Valence, NOT Political Side)

> **Golden Rule:** Label the **emotional tone / affective valence expressed by the commenter**, regardless of whether they are pro-Sara or anti-Sara!

1. **`Positive`** — Praise, admiration, gratitude, hope, relief, encouragement, or warm support toward *anyone* (prosecutors, defense lawyers, senators, VP Sara, or the country).
   - *Examples:* `"Salute to Atty. Diokno, very clear and respectful!"`, `"Praying for truth and justice for the Philippines 🙏"`, `"Good job defense team!"`
2. **`Negative`** — Anger, disgust, frustration, mockery/sarcasm, insults, accusations of corruption/lying, fear, or strong disappointment.
   - *Examples:* `"Mga magnanakaw at sinungaling!"`, `"Walang kwenta talaga si Chiz nakakainis"`, `"Wasting taxpayers' money on this circus"`, `"Wow, 'wala po akong naalala' na naman 🙄"` (sarcastic criticism = `Negative`).
3. **`Neutral`** — Factual observations, timestamps, neutral questions, procedural notes, or ambiguous banter without clear positive/negative emotion.
   - *Examples:* `"What time does Day 45 resume?"`, `"Article II covers unexplained wealth."`, `"Nasa hearing si Atty. Cabrera ngayon."`

### Edge Cases
- **Sarcasm / Irony:** Label the **intended** emotion (e.g., `"Galing magpalusot haha 👏"` mocking someone = `Negative`).
- **Mixed Positive + Negative:** Pick the **dominant** emotion in the comment; if truly 50/50 balanced, choose `Neutral` and note `mixed` in `notes`.
