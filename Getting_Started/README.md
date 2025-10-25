# Ticket Topic Modeling – Technical Deep Dive

This document describes the internal logic, design decisions, and data flow of the ticket topic modeling pipeline.

## 1. High-Level Architecture

Stages:
1. Data Fetch (MSSQL) → Pydantic `Ticket` objects
2. Translation (Azure Translator) with caching
3. Cleaning (Azure OpenAI LLM + deterministic fallback)
4. Product parsing & grouping (optional)
5. Topic Modeling (SentenceTransformer → UMAP → HDBSCAN → BERTopic)
6. Topic Labeling (Azure OpenAI GPT) – optional
7. Artifact generation (visualizations, CSV)
8. Persistence (SQLite: runs, topics, documents, representative_docs)
9. Caching (SQLite + pickle) for incremental reruns

## 2. Data Flow & Objects

Raw SQL rows are mapped into the `Ticket` Pydantic model with fields:
- id, created_at, support_region, body, status, raw_subject, product_context,
  translated_body, preprocessed_body, product_label (derived)

A list of `Ticket` objects moves through translation → cleaning → modeling.
Caching is applied at translation and cleaning steps to avoid recomputation.

## 3. Data Fetching (Pipelines/Data/TicketDataFetcher)

- Uses pyodbc with ODBC Driver 18 (encrypted connection).
- Query: selects ticket text + metadata (view: `zendesk.V_TicketCommentsAndRequests`).
- Pickle cache persists entire result set plus timestamp for TTL-based reuse.
- Fresh fetch only performed if cache missing/expired.

## 4. Translation (app/processing.translate_tickets)

Logic per ticket:
1. Check processed cache (`TicketCache`) – if already translated, reuse.
2. Call Azure Translator `detect_and_translate` (batch of size 1 for simplicity).
3. On error: fallback to original body.
4. Persist `translated_body` back to cache.

Normalization: removal tokens (e.g., `redacted`) applied later during cleaning.

## 5. Cleaning (app/processing.clean_translations)

Hybrid approach:
- If `preprocessed_body` exists in cache: re-strip configured removal tokens and skip LLM.
- Otherwise:
  1. Provide full translated body to `TextCleanserLLM` (Azure OpenAI powered).
  2. On exception (e.g., content filter): use deterministic `_basic_clean` fallback:
     - Remove HTML tags, URLs, emails (replace with `[REDACTED]`), credential-like lines, emojis, common salutations.
  3. Strip tokens via compiled regex union pattern.

Rationale: Ensures resiliency + reproducibility even if LLM is unavailable.

## 6. Product Parsing & Grouping (Optional Per-Product Mode)

Feature toggled via CLI `--per-product` or env `ENABLE_PER_PRODUCT_TOPICS=true`.

Algorithm:
- Derive `product_label` from `product_context` by substring (case-insensitive):
  - ai_plugin → IllustratorPlugin
  - colorcards → ColorCard
  - colorplugin → ColorPlugin
  - colorproof → ColorProof
  - colorserver → ColorServer
  - 3rd_party → ThirdPartyProduct
  - Else → Unclear Product
- Group tickets by `product_label`.
- Drop groups smaller than `MIN_PRODUCT_GROUP_SIZE`.
- Run full modeling sequence per group (independent BERTopic models & runs).

## 7. Text Corpus Construction (app/modeling.build_texts_from_tickets)

Preference order for modeling text: `preprocessed_body` → `translated_body` → `body`.
Empty/whitespace-only documents are discarded.

## 8. Embedding & Dimensionality Reduction

- Sentence embeddings: `all-MiniLM-L6-v2` (fast, 384-dim).
- UMAP:
  - n_neighbors=10, n_components=5, min_dist=0.0, metric=cosine
  - Purpose: compress embeddings to low manifold for clustering stability & speed.

## 9. Clustering (HDBSCAN)

Parameters:
- min_cluster_size=15 (can be tuned for typical ticket volumes)
- metric=euclidean (after UMAP projection)
- cluster_selection_method=eom

Outliers labeled -1 by BERTopic/HDBSCAN are counted but not reduced.

## 10. BERTopic Assembly

- CountVectorizer (stop_words='english') builds term-document matrix.
- ClassTfidfTransformer reweights class-based word distributions.
- KeyBERTInspired representation model aids label generation.
- BERTopic integrates: embeddings + UMAP + HDBSCAN + c-TF-IDF.

## 11. Topic Reduction (Optional)

If `TARGET_N_TOPICS` env is set and fewer than current topics → reduction executed via `topic_model.reduce_topics()`.

## 12. Topic Labeling (generate_topic_labels)

Process per topic:
1. Extract top keywords (default 12) and representative docs (10).
2. Construct prompt with keywords + doc snippets.
3. Azure OpenAI chat completion request (temperature ~0.2).
4. Attempt JSON parse; on failure fallback to keyword-based title.
5. Store `label` + `description` (≤160 chars desired).

Resilience:
- 3 retry attempts with exponential backoff.
- Fallback ensures no topic is left unlabeled.

## 13. Artifacts (apply_labels_and_save_artifacts)

Generated:
- `topic_labels.csv` – topic_id, label, description, top_keywords
- `topic_visual.html` – interactive cluster visualization
- `topic_barchart_labeled.html` – frequency barchart

Custom labels injected into model for consistent visuals.

## 14. Persistence (app/persistence.persist_results)

SQLite schema (core tables):
- runs(run_id, created_at, n_docs, n_topics, n_outliers, embedding_model, umap_params, hdbscan_params)
- topics(run_id, topic_id, label, description, top_keywords, doc_count, product_label)
- documents(run_id, doc_index, ticket_id, topic_id, probability, text, created_at, support_region, status, raw_subject, product_context, body, translated_body, preprocessed_body, product_label)
- representative_docs(run_id, topic_id, rep_index, text)

Design choices:
- Use `INSERT OR REPLACE` for idempotency.
- Truncate long text fields to 4000 chars to bound DB size.
- Adds new columns lazily via `_ensure_documents_columns` for backward compatibility.

## 15. Caching Layers

1. Pickle cache for raw ticket fetch (reduces SQL load)
2. SQLite `TicketCache` for translation + cleaning states
3. Environment variable caps to bound memory/compute

## 16. Configuration Flags (Env Vars Summary)
See `README_usage.md` – includes modeling caps, toggles, artifact paths.

Notable interactions:
- MODEL_CREATED_AFTER overrides MODEL_MAX_TICKETS logic.
- Per-product mode uses `MODEL_MAX_TICKETS_PER_PRODUCT` if present.
- Disabling Azure OpenAI removes labeling + advanced cleaning (fallback still works).

## 17. Edge Cases & Handling
| Scenario | Handling |
|----------|----------|
| Empty ticket body | Skipped silently |
| Translator failure | Original text reused |
| LLM content filter | Deterministic cleaner fallback |
| Insufficient docs (<5) per product | Skip modeling for that product |
| Reduction target > current topics | Reduction skipped |
| Missing env vars for labeling | Labels skipped; pipeline continues |

## 18. Performance Considerations
- Embedding model selection balances speed & semantic quality.
- UMAP + HDBSCAN parameters tuned for moderate-scale (≤10k docs typical).
- Per-product segmentation allows parallelization (future enhancement) and improved topical coherence.
- Retry/backoff avoids rate-limit bursts on Azure OpenAI.

## 19. Future Improvements
- Parallel execution per product group
- Persist model objects per product for incremental updates
- Add evaluation metrics (intra-topic coherence, diversity)
- Optional anonymization layer pre-translation
- API wrapper / service deployment

## 20. Troubleshooting Checklist
1. Verify `.env` values loaded (print selective vars if unsure).
2. Confirm network connectivity to SQL & Azure endpoints.
3. Check `topic_results.db` for populated rows (runs/topics/documents).
4. Inspect logs for labeling parse failures.
5. Validate group sizes when expecting per-product runs.

---
For questions: consult code owners or open an internal issue with logs & environment details.
