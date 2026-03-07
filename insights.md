# Competition Insights

## How this project is documented

| File | Purpose | When to update |
|------|---------|----------------|
| `insights.md` (this file) | Stable findings and current priorities. The source of truth for what we know and what to do next. | When a finding is confirmed or a priority changes. |
| `DIARY.md` | Index of daily entries with LB scores and notebook links. | Every session — add a row with date, one-line summary, notebook links, LB score if submitted. |
| `diary/YYYY-MM-DD.md` | Per-session narrative: what we tried, what we found, what to do next. List notebooks used under a `**Notebooks:**` line at the top. | Every session — create a new file. |
| `notebooks/EDA_general.ipynb` | Persistent EDA reference. Findings go here as properly executed cells with outputs. | When a new structural finding about the data or pipeline is confirmed with code. |
| `notebooks/YYYY-MM-DD.ipynb` | Scratch notebook for the session's experiments. | Each session gets its own file. Don't polish these — they're working logs. |

**When you find something new:**
1. Run it in the session notebook first (scratch).
2. If it holds up, add a proper cell + markdown section to `EDA_general.ipynb`.
3. Add a bullet to the relevant finding in `insights.md`, or create a new numbered finding.
4. Update the priority action list in `insights.md` if priorities shift.
5. Write the diary entry for the day, linking the notebook.

---

Swiss law retrieval. English queries → German corpus. Metric: Macro F1.

---

## Data structure

- **train**: German queries, gold citations in German laws/court
- **val/test**: English queries — cross-lingual retrieval at eval time
- **laws_de.csv**: Swiss law articles (~short, median ~200 chars)
- **court_considerations.csv**: 2.5M rows, pre-chunked court ruling sections

---

## Key findings

### 1. Fixed top-k is fundamentally broken

- Val median gold citations: **22**. Baseline `top_k_final=10` structurally misses most val queries.
- But returning 10 for a 1-citation query → 9 false positives → F1 ~18% even with perfect recall.
- **Adaptive top-k based on score gaps is essential**, not optional.

### 2. Baseline truncates queries way too early

- Baseline cuts queries at 2,000 chars. **22% of train queries are longer** than that.
- bge-m3 supports **8,192 tokens** (~25,000 chars at German char/token ratio).
- The baseline uses <10% of available capacity. Raising the cutoff to ~5,000+ chars is a free win.

### 3. Val citation gap is not explained by query complexity

- Val queries are ~1.6× longer than train — but need **11× more citations**.
- Lexical diversity and explicit law references are essentially identical between splits.
- Val was deliberately curated to be citation-heavy (benchmark stress-testing recall).
- Multi-topic queries are the structural driver: 70% of val queries span 2+ SR law topics vs 9% of train.

### 4. 82% of val citations were never seen as training labels

- For 82% of the articles the model must retrieve at eval, no training query ever had that article as a gold answer.
- The articles **exist in the corpus** (val has 0% missing citations) — but the model was never rewarded for finding them.
- This means the bottleneck is **zero-shot generalization** of the retriever, not just top-k tuning.
- bge-m3's cross-lingual semantic quality (English query → German article, unseen during training) is the core lever.

### 5. Court corpus is mostly noise, and mostly irrelevant

- 17.5% of court rows are <100 chars — procedural one-liners with zero retrieval value. Filter before indexing.
- Only ~1.2% of correct answers are court citations. Skipping dense retrieval on court costs almost nothing in recall.

### 7. Cross-encoder reranker is nearly blind to relevant German docs (critical)

- The pipeline passes English queries directly to `bge-reranker-v2-m3` against German documents.
- Measured on a controlled example: exact-match doc scores **0.022** (pipeline) vs **0.787** (DE query / same doc) vs **0.841** (EN query / EN doc).
- Language penalty is concentrated entirely on the relevant pair — irrelevant docs score near-zero in all conditions. The model doesn't score everything lower; it specifically kills the signal you need.
- Root cause: cross-encoders attend across raw token IDs. "detention" and "Untersuchungshaft" share no token surface form and no trained cross-lingual attention bridge (fine-tuned on English MS-MARCO).
- **Fix: translate queries to German before reranking.** Bi-encoder retrieval is fine with English queries; only the reranker needs German input.

### 6. 28.8% of train labels are missing from the corpus — ignore it

- OR, ZGB, StGB articles were left out of laws_de.csv.
- Val/test have **0% missing** — competition was curated to only reference present articles.
- Training missing rate is a data artifact, not a ceiling on leaderboard score.

---

## Experiment results

| Date | Experiment | LB Score |
|------|-----------|----------|
| 2026-03-01 | Baseline (BM25 + hybrid, laws + court) | 0.01816 |
| 2026-03-07 | Laws-only dense (no BM25, semantic only) | 0.02064 |

---

## Priority action list

1. **Translate queries to German before reranking** — controlled experiment shows severe score suppression on relevant docs (0.787 DE/DE vs 0.022 EN/DE on one toy example); needs validation on real val queries before quantifying impact
2. **Adaptive top-k** — use reranker score gaps to decide the cutoff per query (highest leverage)
3. **Raise query char limit** — from 2,000 to at least 5,000 chars (free, low effort)
4. **Filter short court rows** — drop <100 char rows before indexing (free cleanup)
5. **Cross-lingual retrieval quality** — bge-m3 zero-shot generalization is the actual bottleneck; consider multilingual fine-tuning or better prompting strategies
