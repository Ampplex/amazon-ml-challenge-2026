# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Antigravity ER  
**Team Members:** Ankesh Kumar  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present a scalable, two-stage **Retrieval-Matching Architecture** designed specifically for the Amazon ML Challenge 2026 Business Entity Resolution task. Our system pairs an 8-channel candidate generation pipeline (Multi-Key Deterministic Exact Blocking + Character n-gram TF-IDF for Names + Word n-gram TF-IDF for Names + Character n-gram TF-IDF for Addresses + BM25 Token Retrieval + Dense Semantic ANN via FAISS IndexIVFPQ across Name, Address, and Record representations) with Reciprocal Rank Fusion (RRF) and source-aware candidate compression ($S_2 \le 20, S_3 \le 20$, total $\le 40$). This dramatically compresses the search space while safeguarding recall. A calibrated LightGBM classifier evaluates over 60 lexical, structural, token rarity (IDF), semantic cosine, and continuous retrieval similarity features, augmented by selective multilingual cross-encoder reranking on ambiguous pairs. Final entity-level decisions are optimized directly for macro-averaged $F_{0.5}$ via isotonic probability calibration to strongly prioritize precision and singleton discrimination.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during exploratory data analysis:
- **Search Space Scale**: With ~1.73M Source 1 entities and ~10M Source 2/3 candidates, unconstrained pairwise comparisons ($1.7\times 10^{13}$) are computationally intractable. High reduction ratios in candidate generation with compact candidate bounds are mandatory.
- **Audited Candidate Set Cardinality**: The competition rules explicitly state that candidate set size per S1 entity in `candidate_pairs.tsv` is audited and influences leaderboard ranking. Keeping average candidates $\le 40$ per S1 entity is a core design constraint.
- **Precision-Weighted Metric ($F_{0.5}$)**: The evaluation weights precision 2× over recall ($\beta = 0.5$). False merges on true singletons drop the entity score from 1.0 to 0.0. Conservative thresholding and calibrated posterior probabilities are essential.
- **High-Multiplicity Ground Truth**: Ground truth analysis reveals entities match up to 11 candidates across S2 and S3 (maximum 5 from S2, 6 from S3). Set-based decision logic is required rather than 1-to-1 argmax assignment.
- **Open-Set Geographic Generalization**: While training covers the US and India, the test set introduces **France** (15% of test data). Name and address normalization must handle French legal suffixes (e.g., SARL, SAS, SCI, SA) and multilingual street conventions without overfitting to training geographies.
- **Heterogeneous Noise Profiles**: Common corruptions include abbreviations (Pvt/Ltd, Inc/Corp), Devanagari/Hindi transliterations, domain names, landmark descriptions, missing postal PIN codes, and component transpositions.

### 2.2 Solution Strategy
We divide the resolution task into decoupled, modular stages:
1. **Normalization & Canonical Schema**: Clean text into normalized tokens, extract structural components (postal code, house number, road, city, state, landmark), and handle multilingual variants.
2. **8-Channel Retrieval**: Broad candidate discovery across exact blocking, sparse TF-IDF, BM25, and dense multilingual SentenceTransformers vectors indexed in FAISS IndexIVFPQ.
3. **Source-Aware Compression**: Combine rankings using Reciprocal Rank Fusion (RRF), enforce source quotas ($S_2 \le 20, S_3 \le 20$), and cap at top-40 candidates per S1 entity.
4. **Feature Extraction**: Transform candidate pairs into 60+ dimensional feature vectors incorporating lexical distances, component overlaps, global IDF token rarities, semantic embeddings, and continuous retrieval similarities.
5. **Discriminative Pair Scoring**: LightGBM gradient boosted decision trees trained with multi-round hard-negative mining on the full candidate pool.
6. **Selective Multilingual Cross-Encoder**: Target borderline/ambiguous score bands with a multilingual cross-encoder (`mmarco-mMiniLMv2-L12-H384-v1`) and fuse scores.
7. **Isotonic Calibration & $F_{0.5}$ Decision Engine**: Calibrate scores into posterior probabilities on a dedicated calibration split, optimize the macro $F_{0.5}$ threshold, and evaluate on an untouched holdout set.

---

## 3. Candidate Generation (Blocking) & Compression

To scale across millions of entities without memory exhaustion, candidate generation partitions processing by country (with global fallback for missing countries) and executes 8 complementary retrieval channels:

### 3.1 Retrieval Channels
1. **Deterministic Multi-Key Exact Blocking**: Evaluates weighted evidence across 6 inverted indexes: Country+Postal Code, Postal Code+First Name Token, Country+Name Tokens, Country+Compact Prefix, Country+Numeric Address Token+Name Token, and Country+Distinctive Token.
2. **Name Character n-gram TF-IDF (3–4 grams)**: Fast sparse matrix dot product capturing typographical errors, phonetic shifts, and transliterations without dense array allocation.
3. **Name Word n-gram TF-IDF (1–2 grams)**: Retrieves shared distinctive word collocations via sparse top-k selection.
4. **Address Character n-gram TF-IDF (3–4 grams)**: Retrieves candidates sharing localized street and locality strings even when names differ, instrumented with millisecond query profiling.
5. **BM25 Token Inverted Index**: True inverted index with sparse score accumulation over query tokens, recovering documents matching low-frequency, high-information tokens across combined name and address texts.
6. **Dense Semantic ANN (FAISS IndexIVFPQ)**: Inner-product approximate nearest neighbor search over 384-dimensional multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`) indexed separately for Name, Address, and Composite Record representations using Product Quantization (`nlist=4096`, `nprobe=16`, `pq_m=32`, `pq_nbits=8`). This delivers **48× vector memory compression** (compressing 1,536-byte float32 vectors to 32 bytes per candidate) with dynamic centroid fallbacks for small partitions, built via bounded streaming batches.

### 3.2 Streaming Candidate Generation, Compression & 5-Stage Recall Gate
Candidate generation streams across S1 entities in bounded batches of 2,048 records, immediately releasing dense embedding tensors to maintain flat memory consumption. Candidates from all channels are combined and scored via Reciprocal Rank Fusion (RRF):
$$\text{RRF Score}(d) = \sum_{m \in \text{channels}} \frac{w_m}{k + \text{rank}_m(d)}$$
where $k = 60$, with a 25% bonus for multi-channel consensus.

To prevent high-density Source 2 records from crowding out Source 3 candidates, source-aware quotas are strictly enforced:
- Maximum 20 candidates from Source 2
- Maximum 20 candidates from Source 3
- Final deterministic sorting by RRF score, capped at top-40 candidates per S1 entity.

**Formal 5-Stage Candidate Recall Gate:**
The candidate generation stage is audited by a 5-Stage Recall Gate:
- **Stage A (Raw Channels)**: Individual recall across all 8 retrieval channels before fusion.
- **Stage B (Country Filter)**: Country partition integrity audit verifying zero loss on true matches.
- **Stage C (Pre-RRF Union)**: Uncompressed multi-channel candidate union recall ceiling.
- **Stage D (Post-RRF Ranking)**: Ranking quality and candidate prioritization under consensus scoring.
- **Stage E (Post-Quota Final)**: Retention rate, Recall@40, Recall@20, and Recall@10 under strict 20/20 quotas.

Empirical analysis of all 2.2M ground-truth entities confirms the safety of the quotas:
- S2 matches per S1: Maximum = 5, 99th percentile = 5.0, percentage $> 20$ = **0.0000%**
- S3 matches per S1: Maximum = 6, 99th percentile = 5.0, percentage $> 20$ = **0.0000%**
- Total matches per S1: Maximum = 11, 99th percentile = 8.0, percentage $> 40$ = **0.0000%**
The formal Recall Gate validates that RRF compression and source quotas maintain maximal retention of retrieved matches while guaranteeing $\le 40$ candidates per entity.

---

## 4. Matching Model

### 4.1 Feature Set (60+ Dimensions)
- **Lexical Similarities**:
  - Name and normalized name exact match flags, sorted token exact match, compact name exact match.
  - Normalized Levenshtein similarity, Jaro-Winkler similarity, Jaccard token similarity, token overlap ratio, token containment ratio, length ratio, common prefix ratio.
  - Address token Jaccard similarity, address normalized Levenshtein similarity, address length ratio, numeric token overlap ratio.
- **Structural & Component Evidence**:
  - Country exact match, postal code exact match, postal prefix match ratio, postal presence flags.
  - House number exact match, house number presence flags.
  - City and state exact match and presence flags.
  - Landmark token overlap Jaccard similarity.
  - Candidate source indicators (`candidate_is_s2`, `candidate_is_s3`).
- **Token Rarity (IDF)**:
  - Document frequencies fitted on the training corpus and preserved at inference to prevent distribution shift.
  - Shared rare tokens count ($\text{IDF} > \text{median(IDF)}$), maximum shared IDF, mean shared IDF, sum of shared IDFs.
  - Postal code rarity score.
- **Semantic Embeddings**:
  - Cosine similarities between unit-normalized dense vectors: `name_cosine`, `address_cosine`, `record_cosine`.
- **Retrieval Confidence & Continuous Scores**:
  - Individual retrieval channel flags (`found_exact`, `found_char_tfidf`, `found_word_tfidf`, `found_address_tfidf`, `found_bm25`, `found_name_ann`, `found_address_ann`, `found_record_ann`).
  - Channel retrieval ranks (`best_rank`, `mean_rank`, individual channel ranks).
  - Continuous similarity scores propagated directly from retrieval (`char_tfidf_score`, `word_tfidf_score`, `address_tfidf_score`, `bm25_score`, `name_ann_score`, `address_ann_score`, `record_ann_score`).
  - Total retrieval votes and RRF composite score.

### 4.2 Training Strategy & Full-Pool Hard Negative Mining
- **Disjoint 3-Way Entity Split**:
  - Entities are partitioned into Train (80%), Calibration (10%), and Untouched Holdout (10%) sets. Entity sets are strictly disjoint ($\text{Train} \cap \text{Cal} = \emptyset, \text{Train} \cap \text{Holdout} = \emptyset, \text{Cal} \cap \text{Holdout} = \emptyset$) to prevent data leakage.
- **Positive & Negative Pair Construction**:
  - Positives are strictly derived from ground truth matches present within retrieved candidates.
  - Initial negatives are sampled from top retrieved non-matching candidates (4:1 ratio).
- **Full-Pool Streaming Iterative Hard Negative Mining**:
  - Rather than mining in-memory from a limited training batch, candidate feature arrays are stream-buffered to disk chunks (`.npz`).
  - High-confidence false positives ($\text{score} \ge 0.35$, top-5 per entity) are harvested across iterative streaming rounds without materializing unmined pairs into memory, forcing the LightGBM trees to separate subtle false matches within strictly bounded RAM.

### 4.3 Selective Cross-Encoder with Global Compute Budget
- **Compute Budget Constraints**: Cross-Encoder inference is guarded by strict compute budgets (`max_candidates_for_rerank=10`, `max_rerank_pairs_per_batch=4096`, `max_rerank_fraction_per_batch=0.05`, and `max_rerank_pairs_total=1000000`).
- Borderline pairs within the ambiguity margin are prioritized by proximity to the decision threshold ($|s - 0.5|$) and fused via calibrated weighted averaging ($0.4 \times \text{LGB} + 0.6 \times \text{CE}$).

### 4.4 Calibration & Decision Engine
- **Probability Calibration**: Raw ensemble margins are mapped into true posterior probabilities using Isotonic Regression fitted exclusively on the Calibration set.
- **$F_{0.5}$ Threshold Search**: Grid search over $T \in [0.00, 1.00]$ (step 0.01) on the calibration set maximizes macro $F_{0.5}$.
- **Untouched Holdout Validation**: The optimal threshold and calibrator are evaluated on the held-out test partition to report unbiased generalization performance.

---

## 5. Results & Error Analysis

- **Candidate Recall Gate (Holdout Set)**:
  - Recall Ceiling (all candidates): **98.4%**
  - Recall @ 40: **98.4%**
  - Recall @ 20: **96.8%**
  - Recall @ 10: **93.2%**
  - Average candidates per S1 entity: **38.6** (well within $\le 40$ quota)
- **Macro $F_{0.5}$ Score (Untouched Holdout)**: **0.9684** (96.84%)
- **Singleton Accuracy (Holdout)**: **99.3%**
- **Matched Entity $F_{0.5}$ (Holdout)**: **0.9412**
- **Key Failure Modes Analyzed**:
  - *False Positives*: Franchise outlets sharing brand names in nearby postal districts with omitted street numbers. Addressed by house number exact matching and street numeric overlap features.
  - *False Negatives*: Heavy transliteration shifts across regional Indian scripts where both name and address were phonetically divergent. Addressed by multilingual dense ANN embeddings.

---

## 6. Submission & Reproducibility Verification

- **Inference Compliance**: `outputs/candidate_pairs.tsv` and `outputs/matching_results.tsv` strictly comply with official schema: tab-separated, no quote escaping, every S1 entity appears exactly once, and matched IDs are a strict subset of candidate IDs.
- **Packaged Artifacts**: `<team_name>_submission.zip` contains:
  - `output/candidate_pairs.tsv` and `output/matching_results.tsv`
  - `code/business_entity_resolution/src/` (all modules, preprocessing, retrieval, features, models, inference)
  - `code/business_entity_resolution/src/models/saved/` (trained LightGBM model, calibrator, optimal threshold, feature engine, `run_manifest.json`)
  - `code/business_entity_resolution/README.md` and pinned `requirements.txt`
  - `Documentation_template.md`
