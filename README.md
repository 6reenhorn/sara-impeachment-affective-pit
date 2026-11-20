# Public Affective Analysis of VP Sara Duterte's Impeachment (YouTube PH)

CS412 Affective Computing PIT — Text Mining & Sentiment/Emotion Analysis.

## Pipeline Stages
1. `src/01_mining/` — YouTube comment collection with automatic PII anonymization.
2. `src/02_preprocessing/` — Text cleaning, deduplication, and Taglish/Bisaya normalization.
3. `src/03_modeling/` — Feature extraction and affective classification.

## Data Privacy
Commenter usernames, channel IDs, and profile photo URLs are never stored.
Channel identifiers are converted into one-way SHA-256 hashes (`user_hash`) during collection solely to detect spam/bot repetition.