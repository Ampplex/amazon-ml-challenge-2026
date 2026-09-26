Here is the **complete architecture we converged on**, including the role of every stage, what data flows through it, how training works, how inference works, and why each component exists for this specific dataset.

This should be treated as the **frozen reference architecture**. The remaining work is implementation/scaling, not changing this ML design.

# Amazon ML Challenge 2026 — Final Business Entity Resolution Architecture

## 1. Problem formulation

You have three independent sources:

```text
S1 = reference entities
S2 = noisy candidate source
S3 = noisy candidate source
```

For every S1 entity:

```text
S1 entity
   ↓
find ALL matching S2/S3 entities
   ↓
possible output:
0 matches
1 match
many matches
```

The dataset contains noise such as:

```text
Name:
- typos
- abbreviations
- legal suffix changes
- missing words
- duplicated words
- word-order changes
- alternate/trade names
- multilingual/local-script variations

Address:
- abbreviations
- missing components
- reordered components
- house/plot number changes
- landmarks
- postal-code corruption/missing values
- transliteration/local scripts
```

The real examples you supplied demonstrate why the architecture must use **both name and address evidence**. For example, `Maure Williams Colombier` → `Dréxkor` has almost no useful name overlap, while the address contains strong matching evidence. 

The India example similarly contains duplicated words, missing terms, suffix substitutions, address artifacts, and Bengali script. 

The official challenge also explicitly says exhaustive comparison is not viable and that candidate generation must substantially reduce the search space. 

---

# 2. Complete high-level architecture

```text
                         RAW DATA
                ┌──────────┼──────────┐
                ↓          ↓          ↓
               S1         S2         S3
                │          │          │
                └──────────┼──────────┘
                           ↓
              1. DATA INGESTION / SCHEMA
                           ↓
              2. NORMALIZATION + PARSING
                           ↓
              3. COUNTRY-AWARE PARTITION
                           ↓
            ┌──────────────┴──────────────┐
            │                             │
            │     4. MULTI-CHANNEL        │
            │        RETRIEVAL             │
            │                             │
            ├── A. Exact Blocking         │
            ├── B. Name Char TF-IDF        │
            ├── C. Name Word TF-IDF       │
            ├── D. Address Char TF-IDF    │
            ├── E. BM25                   │
            ├── F. Name Dense ANN         │
            ├── G. Address Dense ANN      │
            └── H. Record Dense ANN       │
                           ↓
                5. CANDIDATE UNION
                           ↓
                6. RRF CANDIDATE FUSION
                           ↓
                  7. SOURCE QUOTAS
                   S2 ≤ 20
                   S3 ≤ 20
                  Total ≤ 40
                           ↓
                candidate_pairs.tsv
                           ↓
               8. PAIR FEATURE ENGINE
                           ↓
             ┌─────────────┼─────────────┐
             ↓             ↓             ↓
         Lexical       Structural      Semantic
             ↓             ↓             ↓
          Rarity       Retrieval       metadata
             └─────────────┼─────────────┘
                           ↓
                9. LIGHTGBM PAIR MODEL
                           ↓
               10. AMBIGUITY DETECTION
                           ↓
               11. SELECTIVE CROSS-
                   ENCODER RERANKING
                           ↓
                12. SCORE FUSION
                           ↓
                13. ISOTONIC CALIBRATION
                           ↓
                14. ENTITY-LEVEL
                    F0.5 THRESHOLD
                           ↓
                    0 / 1 / MANY
                           ↓
             ┌─────────────┴─────────────┐
             ↓                           ↓
   matching_results.tsv         candidate_pairs.tsv
```

The current code implements the core retrieval structure and the 20/20/40 compression design. 

---

# 3. Stage 1 — Data ingestion

## Inputs

```text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
```

and at inference time:

```text
test_source1.tsv
test_source2.tsv
test_source3.tsv
```

Each record has:

```text
entity_id
business_name
business_address
country
```

The source is derived from the ID prefix:

```text
S1-...
S2-...
S3-...
```

The challenge explicitly specifies this schema. 

Internally, records are represented by:

```text
Record
├── record_id
├── source_id
├── original_name
├── normalized_name
├── compact_name
├── name_tokens
├── original_address
├── normalized_address
├── address_tokens
├── country
├── state
├── city
├── postal_code
├── house_number
├── locality
├── landmark
├── numeric_tokens
└── record_text
```

---

# 4. Stage 2 — Normalization and parsing

This stage creates canonical representations without deleting the original values.

## Name normalization

Pipeline:

```text
raw name
  ↓
Unicode normalization
  ↓
lowercase
  ↓
punctuation normalization
  ↓
whitespace normalization
  ↓
tokenization
  ↓
abbreviation/legal suffix expansion
  ↓
normalized_name
  ↓
compact_name
```

Examples:

```text
"Maure Williams Colombier Inc"
→
"maure williams colombier incorporated"
```

and:

```text
"Pvt Ltd"
→
"private limited"
```

The legal suffix dictionary also includes French forms such as SARL, SAS, SCI, SA, etc. 

The system keeps:

```text
original_name
normalized_name
compact_name
name_tokens
```

rather than destroying the original representation.

---

# 5. Address normalization and parsing

Addresses receive both textual normalization and structured extraction.

```text
raw address
   ↓
postal extraction
house-number extraction
landmark extraction
numeric-token extraction
   ↓
Unicode normalization
   ↓
lowercase
   ↓
punctuation normalization
   ↓
abbreviation expansion
   ↓
normalized address
   ↓
city/state extraction
```

Extracted components:

```text
postal_code
house_number
city
state
landmark
numeric_tokens
address_tokens
```

The current parser supports US, India and France patterns, with generic fallback. 

This stage is especially important because your dataset includes cases where the name is heavily corrupted but the address remains usable. 

---

# 6. Stage 3 — Country-aware partitioning

Instead of considering all ~10M candidate records for every S1:

```text
S1(country = US)
        ↓
S2(country = US)
S3(country = US)
```

Similarly:

```text
S1(country = India)
        ↓
S2(country = India)
S3(country = India)
```

and so on.

This is **not** hard-coding:

```text
{US, India}
```

because France exists in the test set. The challenge explicitly requires treating country as an open-set label and ensuring France entities are included. 

For missing country:

```text
unknown S1
    ↓
unknown S2/S3 first
    ↓
fallback if no unknown candidates exist
```

The current code has a shared country-partition helper used by training and inference. 

---

# 7. Stage 4 — Multi-channel retrieval

This is the most important architectural section.

No single retrieval method is trusted to recover every true match.

Instead:

```text
S1
 ↓
8 independent retrieval channels
 ↓
candidate union
```

---

## Channel A — Deterministic exact blocking

The exact blocker contains six inverted indexes:

```text
1. country + postal
2. postal + first name token
3. country + first/second name tokens
4. country + compact name prefix
5. country + numeric token + name token
6. country + distinctive name token
```

Each candidate receives evidence from multiple exact keys.

Example:

```text
same country + same postal
      → strong evidence
```

```text
same country + same distinctive name token
      → weaker evidence
```

This is not just binary exact matching. It accumulates weighted structural evidence.

The implementation currently uses these six indexes. 

### Why it exists

This channel handles:

```text
exact matches
near-exact structural matches
high-confidence matches
```

It gives the system very strong anchors.

---

# 8. Channel B — Name character TF-IDF

Input:

```text
normalized_name
```

Representation:

```text
character n-grams
3–4
```

Example:

```text
"williams"
```

produces character fragments such as:

```text
wil
ill
lli
lia
iam
ams
...
```

This helps recover:

```text
Williams
Wilblims
```

because many character fragments remain shared.

The current implementation performs a sparse matrix multiplication and then selects top-k candidates rather than constructing a dense full similarity matrix. 

---

# 9. Channel C — Name word TF-IDF

Representation:

```text
1–2 word n-grams
```

This captures distinctive lexical combinations.

Example:

```text
"maure williams colombier"
```

can retrieve candidates containing:

```text
maure williams
williams colombier
```

This is complementary to character TF-IDF.

Character TF-IDF handles:

```text
spelling corruption
```

Word TF-IDF handles:

```text
distinctive lexical identity
```

---

# 10. Channel D — Address character TF-IDF

This is particularly important for your dataset.

Input:

```text
normalized_address
```

with character n-grams.

This allows recovery when:

```text
business_name ≈ completely unrelated
```

but:

```text
address ≈ similar with typos
```

Your supplied `Dréxkor` example is exactly why this channel exists. 

---

# 11. Channel E — BM25

BM25 is a token-based retrieval channel over:

```text
name_tokens + address_tokens
```

The implementation uses a true inverted index:

```text
token
 ↓
posting list
 ↓
document IDs + term frequencies
```

Query:

```text
S1 tokens
 ↓
posting lists
 ↓
BM25 accumulation
 ↓
top-k candidates
```

This is useful for rare/distinctive tokens.

For example:

```text
Ticonderoga
```

can carry substantially more identity information than a generic word such as:

```text
company
```

The current BM25 implementation explicitly builds an inverted index and prunes excessively common terms. 

---

# 12. Channel F — Name Dense ANN

The system uses:

```text
paraphrase-multilingual-MiniLM-L12-v2
```

with:

```text
384-dimensional embeddings
```

Input:

```text
normalized_name
```

Index:

```text
FAISS IVFPQ
```

Current configuration:

```text
nlist = 4096
nprobe = 16
pq_m = 32
pq_nbits = 8
```

The latest FAISS implementation is configurable for IVFPQ and uses streaming batches when constructing the index. 

### Why

Dense retrieval helps with:

```text
semantic similarity
cross-lingual similarity
high lexical corruption
alternate naming patterns
```

---

# 13. Channel G — Address Dense ANN

Exactly the same idea, but using:

```text
normalized_address
```

rather than the business name.

This is particularly useful for:

```text
address typos
translated addresses
local-script addresses
reordered components
partial addresses
```

---

# 14. Channel H — Full-record Dense ANN

The third representation is:

```text
normalized_name
+
normalized_address
+
country
```

conceptually:

```text
name | address | country
```

This allows the embedding to see the combined identity signal.

So the three dense representations are deliberately different:

```text
Name embedding
Address embedding
Full-record embedding
```

rather than one embedding doing everything.

---

# 15. Stage 5 — Candidate union

Each retrieval channel produces its own ranked candidates.

Conceptually:

```text
Exact              → candidates
Name Char TF-IDF   → candidates
Name Word TF-IDF   → candidates
Address TF-IDF     → candidates
BM25               → candidates
Name ANN           → candidates
Address ANN        → candidates
Record ANN         → candidates
                    ↓
                 UNION
```

A candidate can appear in multiple channels.

For each candidate, we therefore retain metadata such as:

```text
found_by_exact
found_by_char_tfidf
found_by_word_tfidf
found_by_address_tfidf
found_by_bm25
found_by_name_ann
found_by_address_ann
found_by_record_ann
```

plus:

```text
channel scores
channel ranks
retrieval_votes
```

This metadata is later fed into LightGBM.

---

# 16. Stage 6 — RRF candidate compression

Simply unioning all retrieval results would produce too many candidates.

So you compute:

```text
RRF(d) =
Σ 1 / (k + rank)
```

with:

```text
k = 60
```

Exact blocking receives an additional high-confidence contribution, and candidates discovered by multiple channels receive a consensus boost.

The important idea is:

```text
candidate appearing in many retrieval channels
                    ↓
              stronger evidence
```

while:

```text
candidate appearing once at a weak rank
                    ↓
              weaker evidence
```

The current implementation computes RRF over all channel ranks and applies the multi-channel boost. 

---

# 17. Stage 7 — Source-aware 20/20/40 compression

After RRF:

```text
S2 candidates → independently ranked
S3 candidates → independently ranked
```

then:

```text
S2 → top 20
S3 → top 20
```

followed by:

```text
combined → top 40
```

So:

```text
             Candidate Union
                   ↓
                  RRF
             ↙           ↘
          S2 ranking    S3 ranking
             ↓              ↓
          top 20          top 20
             ↘              ↙
                  merge
                    ↓
                  top 40
```

This protects against one source dominating the candidate budget.

The supplied dataset context shows a maximum of 5 S2 and 6 S3 true matches per S1, so the quota provides substantial capacity above the observed true-match multiplicity. 

---

# 18. `candidate_pairs.tsv`

This is a crucial conceptual point.

`candidate_pairs.tsv` is **not** the raw exact-blocking output.

It must represent:

```text
the exact candidates that the final matching model actually scores
```

That is explicitly required by the challenge. 

So the pipeline is:

```text
raw retrieval
   ↓
RRF
   ↓
20/20
   ↓
40 max
   ↓
candidate_pairs.tsv
   ↓
LightGBM
```

That is the final candidate set.

---

# 19. Stage 8 — Pair Feature Engine

Now every:

```text
(S1, Candidate)
```

pair becomes a feature vector.

The architecture has five feature families.

---

## Family A — Lexical features

Examples:

```text
name_exact
name_normalized_exact
name_jaccard
name_token_overlap
name_token_containment
name_levenshtein
name_jaro_winkler
name_length_ratio
name_common_token_count
name_prefix_match
name_sorted_exact
compact_name_exact
```

Address:

```text
address_jaccard
address_token_overlap
address_levenshtein
address_length_ratio
numeric_overlap
```

These features answer:

> How similar are the two strings structurally?

The implementation calculates these directly between S1 and candidate records. 

---

# 20. Family B — Structural/component features

These include:

```text
country_exact
postal_present_both
postal_exact
postal_prefix_match
house_number_present_both
house_number_exact
city_present_both
city_exact
state_present_both
state_exact
both_name_present
both_address_present
name_len_ratio
address_len_ratio
numeric_count_diff
candidate_is_s2
candidate_is_s3
landmark_overlap
```

These features answer:

> Does the physical/business structure of the records align?

For example:

```text
same postal
+
same house number
+
same city
+
similar name
```

is much stronger than name similarity alone.

---

# 21. Family C — Semantic features

The three semantic features are:

```text
name_cosine
address_cosine
record_cosine
```

They come from the dense representations.

The latest implementation deliberately avoids storing a giant global embedding cache during the normal path and uses the ANN similarity already present on the `CandidatePair`. 

So a candidate can have:

```text
name_cosine = 0.85
address_cosine = 0.25
record_cosine = 0.79
```

which tells LightGBM:

> semantic name/record evidence is strong even though the address isn't.

This is valuable for alternate business names.

---

# 22. Family D — Rarity / IDF features

These features measure how informative shared tokens are.

Conceptually:

```text
common token
→ low IDF
→ weak evidence
```

```text
rare token
→ high IDF
→ strong evidence
```

Features include:

```text
name_rare_token_overlap
name_max_shared_idf
name_mean_shared_idf
name_sum_shared_idf
address_rare_token_overlap
address_max_shared_idf
postal_rarity
```

The candidate corpus is S2+S3, keeping S1 out of candidate IDF estimation.

The current feature engine explicitly has streaming candidate-corpus IDF fitting support. 

---

# 23. Family E — Retrieval metadata features

The model also receives information about **how the candidate was retrieved**.

Examples:

```text
found_exact
found_char_tfidf
found_word_tfidf
found_address_tfidf
found_bm25
found_name_ann
found_address_ann
found_record_ann
```

Continuous:

```text
char_tfidf_score
word_tfidf_score
address_tfidf_score
bm25_score
name_ann_score
address_ann_score
record_ann_score
```

Rank:

```text
name_ann_rank
address_ann_rank
record_ann_rank
char_tfidf_rank
word_tfidf_rank
address_tfidf_rank
bm25_rank
```

Aggregate:

```text
retrieval_votes
rrf_score
best_rank
mean_rank
```

This is important because retrieval itself is evidence.

A candidate retrieved by:

```text
exact
+
BM25
+
address ANN
+
record ANN
```

should mean something different from one retrieved only once by a weak semantic channel.

---

# 24. Stage 9 — LightGBM pair model

Now:

```text
candidate pair
     ↓
60+ features
     ↓
LightGBM
     ↓
pair score
```

The model is a binary classifier:

```text
1 = same entity
0 = different entity
```

The current implementation uses gradient-boosted decision trees with configurable parameters and early stopping. 

### Why LightGBM here?

Entity-resolution features are highly heterogeneous:

```text
exact flags
continuous similarities
counts
ranks
categorical source indicators
IDF
semantic scores
```

Tree models are well suited to interactions such as:

```text
name similarity = 0.8
AND
postal exact = 1
AND
house number exact = 1
```

versus:

```text
name similarity = 0.8
BUT
postal mismatch
AND
address completely different
```

The model learns these combinations rather than relying on one manually defined formula.

---

# 25. Training positives and negatives

Positive pairs are:

```text
ground-truth match
AND
candidate was actually retrieved
```

This is important because the matching model is trained on the distribution of candidates it will actually see at inference.

Initial negatives are non-ground-truth candidates from the retrieved set.

The intended basic ratio is approximately:

```text
1 positive : 4 negatives
```

---

# 26. Stage 10 — Hard-negative mining

After initial LightGBM training:

```text
trained LightGBM
       ↓
score difficult non-matches
       ↓
find false positives with high score
       ↓
select top 5 per S1
       ↓
add to training
       ↓
retrain
```

Repeat for the configured number of rounds.

The point is to teach the model about:

```text
very plausible but wrong businesses
```

rather than easy negatives.

Example:

```text
S1:
ABC Pharmacy
123 Main Street

Candidate A:
ABC Pharmacy
123 Main Street

Candidate B:
ABC Pharmacy
125 Main Street
```

Candidate B is much more useful as a hard negative than:

```text
XYZ Automobile
Mumbai
```

because B teaches the boundary between genuine and false matches.

---

# 27. Stage 11 — Ambiguity detection

LightGBM is the primary matcher.

The cross-encoder is **not** applied to every candidate.

Instead:

```text
LightGBM score
     ↓
Is candidate clearly obvious?
       / \
     yes  no
     ↓     ↓
final   ambiguous
           ↓
      cross-encoder
```

Ambiguity is determined using:

```text
score band
+
margin to top candidate
```

The current configuration uses:

```text
score band: 0.3–0.8
margin threshold: 0.15
max 10 candidates/S1
```

and now also has per-batch and per-run CE budgets. 

---

# 28. Stage 12 — Selective multilingual cross-encoder

For ambiguous pairs:

```text
S1 textual representation
+
candidate textual representation
        ↓
cross-encoder
        ↓
semantic pair score
```

Input is essentially:

```text
S1 normalized name + address
             versus
candidate normalized name + address
```

This model can jointly attend to both records, unlike independent embedding vectors.

That makes it useful when:

```text
"Prabhav Business Center"
        vs
"Mr Prabhav Business Services"
```

requires understanding the relationship among multiple terms rather than simple vector similarity.

The current model is:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

---

# 29. Stage 13 — Score fusion

Now we have:

```text
LightGBM score
Cross-encoder score
```

For CE-selected pairs:

```text
final fused score =
    0.4 × LightGBM
  + 0.6 × Cross-Encoder
```

For candidates not reranked by CE:

```text
fused score = LightGBM score
```

The current `FusionModel` implements this weighted fusion. 

So the CE acts as a **specialist refinement layer**, not the primary global model.

---

# 30. Stage 14 — Isotonic calibration

The fused score is not automatically a calibrated probability.

So:

```text
raw fused score
      ↓
Isotonic Regression
      ↓
calibrated probability
```

For example:

```text
raw = 0.72
```

might become:

```text
calibrated = 0.91
```

or perhaps:

```text
calibrated = 0.64
```

depending on observed calibration behavior.

The point is that the final decision threshold can operate on a more meaningful probability-like scale.

The current implementation supports isotonic calibration and Platt calibration. 

---

# 31. Stage 15 — Entity-level F0.5 decision

This is one of the most important parts.

You are **not** doing:

```text
argmax(candidate score)
```

because an S1 can have:

```text
0
1
many
```

true matches.

Instead:

```text
S1
 ↓
all candidate calibrated scores
 ↓
keep candidates with:
score >= threshold
 ↓
0 / 1 / MANY
```

The threshold is optimized directly against macro F0.5.

The challenge defines F0.5 as precision-heavy and explicitly gives singleton entities full credit when the prediction is correctly empty. 

The current threshold optimizer searches the configured calibrated-score range and maximizes macro F0.5 on calibration entities. 

---

# 32. Why entity-level thresholding matters

Suppose:

```text
S1-A
candidate 1 → .96
candidate 2 → .91
candidate 3 → .04
```

threshold:

```text
0.80
```

gives:

```text
candidate 1
candidate 2
```

which allows MANY.

Now suppose:

```text
S1-B
candidate 1 → .38
candidate 2 → .22
```

with threshold:

```text
0.80
```

you correctly output:

```text
[]
```

This is how the system handles singletons.

---

# 33. Training architecture

The complete training flow is:

```text
Training S1/S2/S3
       ↓
Normalization
       ↓
Create entity-disjoint split
       ↓
80% Train
10% Calibration
10% Holdout
       ↓
Global S2/S3 rarity statistics
       ↓
Country partition
       ↓
Build retrieval indexes
       ↓
Candidate retrieval
       ↓
RRF + 20/20/40
       ↓
Generate training features
       ↓
LightGBM
       ↓
Streaming hard-negative mining
       ↓
LightGBM retraining
       ↓
Calibration candidates
       ↓
Selective CE
       ↓
Fusion
       ↓
Isotonic calibration
       ↓
F0.5 threshold optimization
       ↓
Frozen:
  LightGBM
  calibration model
  threshold
```

Then holdout is evaluated only after these parameters are fixed.

---

# 34. Train / calibration / holdout separation

You intentionally have:

```text
Train
 └─ model learning
 └─ hard negatives

Calibration
 └─ isotonic calibration
 └─ threshold optimization

Holdout
 └─ final evaluation only
```

Conceptually:

```text
TRAIN
   ↓
learn model

CAL
   ↓
learn probability mapping
   ↓
learn decision threshold

HOLDOUT
   ↓
measure final performance
```

This prevents the final threshold from simply being tuned to the holdout data.

---

# 35. Candidate recall gate

Before trusting LightGBM, the architecture checks:

> Was the true candidate even retrieved?

Because:

```text
true match not retrieved
        ↓
LightGBM can never recover it
```

Therefore candidate recall is the ceiling on downstream recall.

The audit should track:

```text
Stage A: raw retrieval channels
Stage B: after country filtering
Stage C: pre-RRF union
Stage D: after RRF
Stage E: after 20/20/40 compression
```

And measure:

```text
Recall@10
Recall@20
Recall@40
Recall-ALL
Recall-ANY
```

plus per-channel recovery.

This is particularly important because your architecture deliberately compresses millions of possible records to at most 40 candidates per S1.

---

# 36. Inference architecture

The final inference flow is:

```text
test_source1
test_source2
test_source3
        ↓
identify countries
        ↓
country partition
        ↓
load candidate partition
        ↓
normalize
        ↓
build retrieval indexes
        ↓
S1 batch
        ↓
8-channel retrieval
        ↓
RRF
        ↓
20/20/40
        ↓
PairFeatures
        ↓
LightGBM
        ↓
Ambiguity detector
        ↓
Selective CE
        ↓
Fusion
        ↓
Calibration
        ↓
Threshold
        ↓
0 / 1 / MANY
```

The current inference implementation processes S1 in batches and writes results incrementally rather than waiting to construct the final output in RAM. 

---

# 37. Final outputs

Two files:

```text
candidate_pairs.tsv
matching_results.tsv
```

For every S1:

```text
S1-ID    candidate IDs
```

and:

```text
S1-ID    final matched IDs
```

The challenge requires exactly one row for every test S1 entity, no duplicate IDs, and only valid S2/S3 IDs. 

And:

```text
matching_results ⊆ candidate_pairs
```

must hold. 

---

# 38. Full architecture in one technical diagram

```text
                              ┌─────────────────────────┐
                              │     S1 / S2 / S3 RAW    │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │  Schema + Data Ingestion │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ Normalization + Parsing  │
                              │                         │
                              │ Name                    │
                              │ Address                 │
                              │ Postal / House / City   │
                              │ State / Landmark        │
                              │ Numeric tokens          │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ Country-aware partition │
                              │ + unknown fallback      │
                              └────────────┬────────────┘
                                           │
                 ┌─────────────────────────┼────────────────────────┐
                 │                         │                        │
                 ▼                         ▼                        ▼
        Exact Blocking              Sparse Retrieval          Dense Retrieval
        │                           │                          │
        ├─ country+postal           ├─ name char TF-IDF       ├─ name ANN
        ├─ postal+name              ├─ name word TF-IDF       ├─ address ANN
        ├─ country+tokens           ├─ address char TF-IDF    └─ record ANN
        ├─ compact prefix           └─ BM25
        ├─ numeric+name
        └─ distinctive token
                 │                         │                        │
                 └─────────────────────────┼────────────────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │    Candidate Union      │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │       RRF Fusion        │
                              │     k = 60 + consensus  │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │ Source-aware Compression│
                              │       S2 ≤ 20            │
                              │       S3 ≤ 20            │
                              │       Total ≤ 40         │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │  candidate_pairs.tsv    │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │    Pair Feature Engine  │
                              │                         │
                              │ Lexical                  │
                              │ Structural               │
                              │ Semantic                 │
                              │ Rarity / IDF             │
                              │ Retrieval metadata       │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │       LightGBM           │
                              │    Primary pair model    │
                              └────────────┬────────────┘
                                           ▼
                              ┌─────────────────────────┐
                              │  Ambiguity Detection     │
                              │ score + margin + budget  │
                              └────────────┬────────────┘
                                           │
                              ┌────────────┴────────────┐
                              │                         │
                             clear                   ambiguous
                              │                         │
                              │                         ▼
                              │             ┌─────────────────────┐
                              │             │ Selective Cross-     │
                              │             │ Encoder               │
                              │             └──────────┬──────────┘
                              │                        │
                              └────────────────────────┤
                                                       ▼
                                          ┌─────────────────────────┐
                                          │      Score Fusion       │
                                          │  LGBM 0.4 + CE 0.6      │
                                          └────────────┬────────────┘
                                                       ▼
                                          ┌─────────────────────────┐
                                          │   Isotonic Calibration  │
                                          └────────────┬────────────┘
                                                       ▼
                                          ┌─────────────────────────┐
                                          │ Entity-level F0.5       │
                                          │ threshold optimization  │
                                          └────────────┬────────────┘
                                                       ▼
                                          ┌─────────────────────────┐
                                          │       0 / 1 / MANY      │
                                          └────────────┬────────────┘
                                                       ▼
                                          ┌─────────────────────────┐
                                          │ matching_results.tsv    │
                                          └─────────────────────────┘
```

---

# 39. The engineering layer around this architecture

This sits **underneath** the ML architecture and should not change the model design:

```text
Persistent/disk-backed candidate indexes
        ↓
Streaming candidate batches
        ↓
Streaming feature chunks
        ↓
Disk-backed hard-negative mining
        ↓
Memory-bounded LightGBM training
        ↓
Memory-bounded inference
```

That is the part we've been fixing recently.

So the distinction is:

```text
                 CORE ML ARCHITECTURE
                       ✅ FROZEN

8 retrieval
→ RRF
→ 20/20/40
→ features
→ LightGBM
→ selective CE
→ fusion
→ calibration
→ F0.5
```

versus:

```text
                 EXECUTION ENGINE
                       ⚠️ STILL
                    BEING OPTIMIZED

how those millions of records,
vectors, candidates and features
are stored and streamed
```

---

# 40. What we deliberately did NOT add

The following are **not** part of the core architecture:

```text
❌ External business lookup
❌ Geocoding
❌ Government/business databases
❌ Another LLM
❌ Large generative model
❌ Graph neural network
❌ Graph matching as primary resolver
❌ One-to-one assignment
❌ Exhaustive pairwise comparison
```

Graph/triangle consistency was discussed as an optional future enhancement, but we deliberately did **not** make it a dependency of the main system.

The challenge explicitly prohibits external business-data lookup and geocoding, so keeping the system entirely within the provided data is also important. 

---

## Final frozen reference

The architecture we agreed on is therefore:

**Retrieval-first, multi-channel, precision-oriented entity resolution:**

```text
Normalize
→ Country Partition
→ 8-channel Retrieval
→ Candidate Union
→ RRF
→ 20 S2 / 20 S3 / 40 Total
→ 60+ Pair Features
→ LightGBM
→ Selective Multilingual Cross-Encoder
→ Weighted Fusion
→ Isotonic Calibration
→ Entity-level F0.5 Threshold
→ 0 / 1 / MANY
→ Submission Files
```

That is the architecture I would use as the **single reference diagram** for your codebase, documentation, PPT, and coding-agent instructions.