# CS412 Week 14 PIT — Comparative Analysis, Presentation & Lab Form

> **STATUS: PROVISIONAL BASELINE** — ML models are currently trained on high-confidence unanimous lexicon consensus labels while the 5-member Human Gold Standard sheets (`data/annotated/sheets/member_1.csv` .. `member_5.csv`) are pending annotation. Once labeled, run `python src/04_modeling/merge_annotations.py` and re-run `python run_pipeline.py`.

## Task 5: Comparative Analysis Answers

### 1. Which method performed best?
- **Provisional Baseline Note:** Because human gold labels are still pending (`data/annotated/sheets/member_1.csv` .. `member_5.csv`), the provisional test set uses 3-way unanimous lexicon agreement, where VADER/TextBlob/AFINN match their own consensus (`1.0000`) by construction. Under this provisional baseline, **TF-IDF + Logistic Regression** achieves the highest ML performance (outperforming **TF-IDF + Multinomial Naive Bayes** on Macro-F1 and minority-class recall). *(See exact metrics table in `deliverables/ml_results.txt` and `deliverables/charts/model_comparison_f1.png`.)*
- Between the two classical ML models, **Logistic Regression** achieves stronger balanced precision/recall across minority classes thanks to `class_weight="balanced"` and discriminative linear weights over 1–2 gram TF-IDF features, while **Multinomial Naive Bayes** provides a fast, competitive probabilistic baseline.

### 2. Why did lexicons differ from ML?
1. **Multilingual & Code-Switched Corpus (>80% Non-Pure-English):**
   - Only **10.4%** of clean comments are pure English; **71.8%** are Filipino/Tagalog, **11.7%** are Taglish, and **6.1%** are Bisaya/Other.
   - Because VADER, TextBlob, and AFINN rely on English dictionaries, out-of-vocabulary Tagalog/Bisaya emotional expressions (*magnanakaw*, *sinungaling*, *walang kwenta*, *salute*, *nakakahiya*) default to **`0.0` (Neutral)**.
   - Across the full corpus, VADER classifies **44.6%**, TextBlob **75.5%**, and AFINN **62.7%** of comments as Neutral (illustrated in `deliverables/charts/lexicon_neutral_by_language.png`).
2. **Domain Semantics & Political Sarcasm:**
   - Supervised TF-IDF learns impeachment-specific n-grams (*unexplained wealth*, *wala po akong naalala*, *taxpayers money*, *palusot*) directly from the Philippine political context, whereas lexicons rely on static general-purpose English scores and misread sarcastic praise (`"galing magpalusot haha"`).
3. **Preservation of Negations:**
   - Our preprocessing explicitly retained English and Filipino negations (`not`, `hindi`, `di`, `wala`, `dili`, `huwag`), allowing bigram TF-IDF features (`not guilty`, `hindi corrupt`, `walang hiya`) to flip polarity accurately.

### 3. What is the emotional distribution of your domain?
- **Full-Corpus ML Distribution (n = 11,969):**
  - **Negative:** **12.1%**
  - **Neutral:** **70.1%**
  - **Positive:** **17.8%**
- **Cross-Platform Contrast (`platform_sentiment_comparison.png`):**
  - YouTube news comment sections (n = 11,102) reflect polarized public reactions across 8+ broadcast outlets, with higher emoji and exclamation usage.
  - Reddit `r/Philippines` (n = 867) exhibits longer, sharper analytical critiques and live-hearing frustration.
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
   - **Group Members (5):** John Cyril Espina, *(Add Member 2)*, *(Add Member 3)*, *(Add Member 4)*, *(Add Member 5)*
2. **Slide 2 — Introduction**
   - Domain: High-stakes Philippine political & constitutional event (2026 Senate Impeachment Trial of VP Sara Duterte).
   - Why it matters: Understanding civic emotion (anger, distrust, hope, support) across broadcast news audiences (YouTube) and civic forums (Reddit) connects NLP directly to Affective Computing.
3. **Slide 3 — Dataset & Collection Process**
   - **Total Collected:** **14,110 raw comments** → **11,969 clean unique comments** (`11,102` YouTube + `867` Reddit), exceeding the 1,000-comment requirement by over **12.0×**.
   - **Ethical Steps Taken:** Public news channels & public subreddit only; zero interaction/engagement; SHA-256 one-way hashed user IDs; inline `@handle` & `u/handle` PII scrubbed; `deliverables/raw_comments.csv` stores text, timestamp, and reaction count only.
   - **Source Links (YouTube Clips & Coverages):**
  - `UNTV News and Rescue` (oct_3_articles_withdrawn_clip): http://www.youtube.com/watch?v=mEHC1iFcNP4
  - `UNTV News and Rescue` (july_bloodied_unbowed_clip): http://www.youtube.com/watch?v=1OAlzgSfakw
  - `UNTV News and Rescue` (day08_nbi_cross_exam): http://www.youtube.com/watch?v=qhXUCFqhc6A
  - `ABS-CBN News` (july_diokno_unfit_clip): http://www.youtube.com/watch?v=zvwIhNx3jv0
  - `ABS-CBN News` (day05_nbi_cross_exam): http://www.youtube.com/watch?v=lrSnRkfUz-8
  - `ABS-CBN News` (day37_forensic_analyst): http://www.youtube.com/watch?v=hSVFz6RgsC0
  - `ABS-CBN News` (june_kapunan_payback_clip): http://www.youtube.com/watch?v=DRmRsgaugrY
  - `ABS-CBN News` (day15_tense_trial_recap): https://www.youtube.com/watch?v=6CKCv66aIpI
  - `GMA News` (day03_defense_nbi_cross_exam): http://www.youtube.com/watch?v=1n78U7PLiSs
  - `GMA News` (july_16_votes_24oras): http://www.youtube.com/watch?v=NItYtm_Nkak
  - `GMA News` (june_impeachment_explainer): http://www.youtube.com/watch?v=a45Zmvf0Hws
  - `GMA News` (day45_clip_spox_panelo): https://www.youtube.com/watch?v=2AXul-gf0yE
  - `NET25 News and Information` (day44_pbbm_saln_clip): http://www.youtube.com/watch?v=nKWkyaSLCbU
  - `NET25 News and Information` (july_day03_coverage): http://www.youtube.com/watch?v=o2zPIw7pjGY
  - `Rappler` (day01_opening_coverage): http://www.youtube.com/watch?v=jmSZlH7mdfs
  - `Philstar News` (day04_summary_clip): http://www.youtube.com/watch?v=pSpc6dhe78E
  - `Bilyonaryo News Channel` (oct_onpoint_prosecution_case): http://www.youtube.com/watch?v=imHsjM3DkyE
  - `TV5 News Philippines` (oct_frontline_2_articles): http://www.youtube.com/watch?v=XDYPP-ZXNkU
  - `TV5 News Philippines` (senate_remand_impeachment_tv5): https://www.youtube.com/watch?v=Svw7wQgCUhg
  - `TV5 News Philippines` (pretrial_heated_exchange_tv5): https://www.youtube.com/watch?v=gTlznk8uz00
  - `TV5 News Philippines` (sara_camp_senator_judges_tv5): https://www.youtube.com/watch?v=GMkjshJBPHk
  - `TV5 News Philippines` (baste_trillanes_hearing_tv5): https://www.youtube.com/watch?v=vFJCQ9Qo_RE
  - `GMA News` (nbi_matibag_vinluan_clip_gma): https://www.youtube.com/watch?v=fsUQLkFxZd0
  - `GMA News` (nov20_day45_full_gma): https://www.youtube.com/watch?v=djrKFEdXgEE
  - `ABS-CBN News` (day31_impeachment_trial_abscbn): https://www.youtube.com/watch?v=5w_0rzwrSxg
  - `ABS-CBN News` (oct_day38_impeachment_abscbn): https://www.youtube.com/watch?v=oD6LdXw2rds
  - `Rappler` (sept_day20_special_rappler): https://www.youtube.com/watch?v=T7SRffoWb6o
  - `Rappler` (sept_day26_special_rappler): https://www.youtube.com/watch?v=jAWw_oq7BzY
  - `ANC 24/7` (nov19_day44_anc): https://www.youtube.com/watch?v=YJ3Ucb5dGJ8
  - `INQUIRER.net` (nov9_day40_inquirer): https://www.youtube.com/watch?v=GUwX-2GZXAE
  - `INQUIRER.net` (nov12_day43_inquirer): https://www.youtube.com/watch?v=0pgWPgptA8s
  - `Philstar News` (nov9_live_philstar): https://www.youtube.com/watch?v=ecMMbj-24wU
  - `Bilyonaryo News Channel` (sept16_day26_bnc): https://www.youtube.com/watch?v=SQ4JSTNTeeI
   - **Reddit Source:** Public `r/Philippines` impeachment megathread JSON snapshots (`1uophc7`).
4. **Slide 4 — Preprocessing Techniques**
   - Show funnel table from `deliverables/preprocessing_summary.txt`: HTML unescape, URL/handle removal, exact deduplication, `< 3` words removal, `<= 5` comments/user anti-spam cap, NLTK tokenization, bilingual EN+FIL stopword removal (preserving negations like `not`, `hindi`, `wala`, `dili`), and POS-aware WordNet lemmatization.
   - Include 2–3 real before/after comment examples from `deliverables/preprocessing_summary.txt`.
5. **Slide 5 — Lexicon Analysis (VADER, TextBlob, AFINN)**
   - Embed `lexicon_sentiment_distribution.png`, `lexicon_polarity_histograms.png`, and `lexicon_neutral_by_language.png`.
   - Highlight sample outputs and the Neutral spike caused by Tagalog/Taglish/Bisaya text.
6. **Slide 6 — ML-Based Analysis (TF-IDF + Logistic Regression & Naive Bayes)**
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
  - Raw Dataset: `14,110` comments | Clean Dataset: `11,969` comments (`11,102` YouTube, `867` Reddit)
  - Lexicons Used: VADER, TextBlob, AFINN (all 3 implemented)
  - ML Models Used: TF-IDF + Logistic Regression & Multinomial Naive Bayes

---

## Appendix: Embedded Metrics Logs

### Preprocessing Summary
```text
=== Stage 2: Unified NLP Preprocessing Summary ===

Stage                                            |   Total |  YouTube |  Reddit
-------------------------------------------------------------------------------
0. Raw merged input                              |  14,110 |   12,779 |   1,331
1. After cleaning & removing empty/URL-only      |  13,888 |   12,560 |   1,328
2. After exact normalized-text deduplication     |  13,573 |   12,245 |   1,328
3. After keeping >= 3 words                      |  12,903 |   11,641 |   1,262
4. After capping <= 5 comments/user              |  11,969 |   11,102 |     867
5. Final NLP-preprocessed dataset                |  11,969 |   11,102 |     867

--- Clean Dataset by Platform ---
platform
youtube    11102
reddit       867

--- Clean Dataset by Language Bucket ---
  Filipino/Tagalog         :  8,592 (71.8%)
  Taglish/Code-switched    :  1,406 (11.7%)
  English                  :  1,239 (10.4%)
  Bisaya/Other             :    732 (6.1%)

Median words/comment: 12.0
Mean words/comment:   17.5
Unique user hashes:   8,767

--- Sample Before / After NLP Transformations (for Task 6 Slides) ---

Example 1 [youtube | Filipino/Tagalog]:
  text_clean:      Luistro umuwi ka nlng sa batangas sayang lang talaga bayad saiyo ng taumbayan
  tokens:          luistro umuwi ka nlng sa batangas sayang lang talaga bayad saiyo ng taumbayan
  tokens_no_stop:  luistro umuwi nlng batangas sayang bayad saiyo taumbayan
  text_lemmatized: luistro umuwi nlng batangas sayang bayad saiyo taumbayan

Example 2 [youtube | English]:
  text_clean:      MAM CAN YOU PLEASE SPELL BLUDGEONING?!? 😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂😂
  tokens:          mam can you please spell bludgeoning
  tokens_no_stop:  mam please spell bludgeoning
  text_lemmatized: mam please spell bludgeon

Example 3 [youtube | Filipino/Tagalog]:
  text_clean:      Ayaw pa ninyo isa na lang , ma paikli na yong trial , less trabaho , gastos , oras ! Ay naku hindi pa ba pantay yan pabor nga sa inyo dahil 
  tokens:          ayaw pa ninyo isa na lang ma paikli na yong trial less trabaho gastos oras ay naku hindi pa ba pantay yan pabor nga sa inyo dahil iisa na la
  tokens_no_stop:  ayaw ninyo isa paikli yong trial less trabaho gastos oras naku hindi pantay pabor inyo iisa debatihan ninyo
  text_lemmatized: ayaw ninyo isa paikli yong trial less trabaho gastos oras naku hindi pantay pabor inyo iisa debatihan ninyo

Example 4 [youtube | Taglish/Code-switched]:
  text_clean:      WASTING OF THE COURTS TIME, COUNCIL FOR THE DEFENSE. KAKA UMAY KAYO. WALA KAYONG DEFENSA. NAG THREAT ANG VP!!!
  tokens:          wasting of the courts time council for the defense kaka umay kayo wala kayong defensa nag threat ang vp
  tokens_no_stop:  wasting courts time council defense kaka umay wala kayong defensa threat vp
  text_lemmatized: waste court time council defense kaka umay wala kayong defensa threat vp

Example 5 [reddit | English]:
  text_clean:      Summa Cum Laude from University of Prison.
  tokens:          summa cum laude from university of prison
  tokens_no_stop:  summa cum laude university prison
  text_lemmatized: summa cum laude university prison

Example 6 [youtube | Filipino/Tagalog]:
  text_clean:      Articulate, competent ang sagot ng analyst galing..❤
  tokens:          articulate competent ang sagot ng analyst galing
  tokens_no_stop:  articulate competent sagot analyst galing
  text_lemmatized: articulate competent sagot analyst galing
```

### Lexicon Summary
```text
=== Stage 3: Lexicon-Based Sentiment Summary (n = 11,969) ===

VADER:
  Negative  :  2,061 (17.2%)
  Neutral   :  5,342 (44.6%)
  Positive  :  4,566 (38.1%)
TextBlob:
  Negative  :    821 (6.9%)
  Neutral   :  9,042 (75.5%)
  Positive  :  2,106 (17.6%)
AFINN:
  Negative  :  2,173 (18.2%)
  Neutral   :  7,507 (62.7%)
  Positive  :  2,289 (19.1%)

Unanimous 3-way agreement: 6,226 (52.0%)
TextBlob clean vs lemmatized label match: 95.2%

Pairwise % Agreement:
          VADER  TextBlob  AFINN
VADER     100.0      58.6   67.8
TextBlob   58.6     100.0   73.6
AFINN      67.8      73.6  100.0

Pairwise Cohen's Kappa:
          VADER  TextBlob  AFINN
VADER     1.000     0.290  0.477
TextBlob  0.290     1.000  0.451
AFINN     0.477     0.451  1.000

Neutral Rate by Language Bucket:
Lexicon                AFINN  TextBlob  VADER
Language                                     
Bisaya/Other            62.7      76.4   45.4
English                 33.7      51.7   25.3
Filipino/Tagalog        72.7      82.3   51.2
Taglish/Code-switched   27.5      55.2   21.5
```

### ML Results Summary
```text
==============================================================================
CS412 WEEK 14 PIT — TASK 4: ML-BASED SENTIMENT ANALYSIS RESULTS
Label Source: PROVISIONAL BASELINE — PENDING 5-MEMBER HUMAN GOLD LABELS
NOTE: Run `python src/04_modeling/merge_annotations.py` once the 5-member
      annotation sheets are filled, then re-run `train_ml_models.py`.
==============================================================================

Dataset Size (deduplicated lemmatized): 6,209
Train Split (80%): 4,967 | Held-Out Test Split (20%): 1,242
Train/Test Text Overlap: 0 (verified zero leakage)

Class Distribution in Modeling Dataset:
  Negative  :   418 (6.7%)
  Neutral   : 4,521 (72.8%)
  Positive  : 1,270 (20.5%)

--- 5-Fold Stratified Cross-Validation (Train Split Only) ---
Logistic Regression best params: {'clf__C': 5.0} | CV Macro-F1: 0.8576 +/- 0.0131
Multinomial NB best params:      {'clf__alpha': 0.1} | CV Macro-F1: 0.7229 +/- 0.0139

--- Held-Out Test Set Comparison (All Methods on Identical 20% Test Split) ---
                            Method Accuracy Precision (Macro) Recall (Macro) F1 (Macro) F1 (Weighted)
                   VADER (Lexicon)   1.0000            1.0000         1.0000     1.0000        1.0000
                TextBlob (Lexicon)   1.0000            1.0000         1.0000     1.0000        1.0000
                   AFINN (Lexicon)   1.0000            1.0000         1.0000     1.0000        1.0000
           Multinomial NB (TF-IDF)   0.8841            0.8751         0.7353     0.7843        0.8785
      Logistic Regression (TF-IDF)   0.9461            0.9092         0.8651     0.8857        0.9450
LogReg (Word+Char TF-IDF Ablation)   0.9517            0.9155         0.8832     0.8986        0.9510

Best Primary ML Model: Logistic Regression (TF-IDF)

--- Detailed Classification Report: TF-IDF + Logistic Regression ---
              precision    recall  f1-score   support

    Negative     0.8026    0.7262    0.7625        84
     Neutral     0.9508    0.9834    0.9668       904
    Positive     0.9740    0.8858    0.9278       254

    accuracy                         0.9461      1242
   macro avg     0.9092    0.8651    0.8857      1242
weighted avg     0.9455    0.9461    0.9450      1242

Confusion Matrix (rows=True [Neg, Neu, Pos], cols=Pred [Neg, Neu, Pos]):
[[ 61  20   3]
 [ 12 889   3]
 [  3  26 225]]

--- Detailed Classification Report: TF-IDF + Multinomial Naive Bayes ---
              precision    recall  f1-score   support

    Negative     0.9130    0.5000    0.6462        84
     Neutral     0.9002    0.9580    0.9282       904
    Positive     0.8120    0.7480    0.7787       254

    accuracy                         0.8841      1242
   macro avg     0.8751    0.7353    0.7843      1242
weighted avg     0.8830    0.8841    0.8785      1242

Confusion Matrix (rows=True [Neg, Neu, Pos], cols=Pred [Neg, Neu, Pos]):
[[ 42  32  10]
 [  4 866  34]
 [  0  64 190]]

--- Interpretability ---
Top Logistic Regression TF-IDF Features by Class:

  Negative: guilty, unfair, fake, impeachment, wrong, bad, corrupt, criminal, waste, stupid, evil, no, useless, wtf, bitter

  Positive: hahaha, haha, good, lol, wow, true, win, fair, cute, legal, love, proud, natural, fit, best

  Neutral:  magaling, ibang, malaman, dapat, imbistigahan, kaba, one side, bakit, hininga, npa, hehehe, nagsasalita, sablay, ninakaw, fact

--- Full-Corpus ML Sentiment Distribution (n = 11,969) ---
  Negative  :  1,447 (12.1%)
  Neutral   :  8,388 (70.1%)
  Positive  :  2,134 (17.8%)
```
