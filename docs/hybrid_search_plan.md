# Plan: hybrid search (dense vectors + BM25, combined with RRF)

Status: **built** (2026-11-18): migration `0005`, `INDEX_VERSION = 4`, `mode` = hybrid / dense / sparse.

First eval (golden v3, 24 answerable questions, top_k 5):

| mode | hit@5 | recall@5 | precision@5 | MRR |
|---|---|---|---|---|
| dense | 1.00 | **0.94** | 0.50 | **0.97** |
| sparse | 0.88 | 0.82 | 0.46 | 0.85 |
| hybrid (plain RRF) | 1.00 | 0.93 | 0.50 | 0.93 |
| **hybrid (weighted RRF, dense 2 : bm25 1)** — in use | 1.00 | **0.95** | **0.51** | 0.95 |

Plain RRF hurt `education` (q18 "electrical engineering degree": BM25 can't match "EEE"). Weighted RRF
(`RRF_WEIGHTS` in `retriever_service.py`) fixed that: hybrid now beats dense on recall and precision, and no
type has lower recall. Still open: q10 ("RAG" ≠ "RAGAS" for BM25), and education MRR (0.42 vs 0.67 dense:
the right chunk is found but at #2–#3).

---

## 1. Why we need it

Our eval (INDEX_VERSION 3, top_k 5) still misses chunks that contain the **exact words** of the question:

| Question | Missed chunk | The exact word is right there |
|---|---|---|
| q04 "Who has worked with RabbitMQ or message queues?" | `Skills - Core Technical Skills: … Message Queues` | "Message Queues" |
| q10 "Who has measured the quality of a RAG system?" | `Skills - LLMOps/Evaluation: … Guardrails, RAGAS` | "RAGAS" |

Today we only have **dense search** (meaning). It is great at *"suggests movies to users" → Recommendation Engine*,
but weak at exact names like `RabbitMQ`, `RAGAS`, `SpiceDB`, `LightFM`. Resumes are full of exact names.

**Hybrid search = run both kinds of search and merge the results**:

| | Dense (what we have) | BM25 / keyword (what we add) |
|---|---|---|
| Finds | same *meaning*, different words | same *words* |
| Good at | "message queues" ↔ "async messaging" | "RAGAS" ↔ "RAGAS" |
| Bad at | rare exact names, acronyms | synonyms, paraphrases |

They fail on different questions, so together they miss less.

---

## 2. The big picture

```
                         INDEXING (upload / rebuild)
chunk text ──┬── HF bge-base-en-v1.5 ──► dense vector  [0.12, -0.03, ... 768 numbers]
             └── fastembed Qdrant/bm25 ► sparse vector {"rabbitmq": 1.4, "messag": 0.9, ...}
                                              │
                         both stored in ONE Qdrant point (same id, same payload)


                         SEARCH (one request to Qdrant)
question ──┬── dense query vector ──► Qdrant: top 20 by meaning   ──┐
           │                                (filtered by user)       ├── Qdrant merges with weighted RRF ──► top 5
           └── sparse query vector ─► Qdrant: top 20 by keywords  ──┘        (dense counts 2x, BM25 1x)
```

**Qdrant does all the searching and the merging.** Our code only makes the two query vectors and sends one request.

---

## 3. Dense vs BM25 in simple words

### Dense (meaning)

The whole chunk becomes 768 numbers. Similar meaning → similar numbers.

```
"async messaging (RabbitMQ)"    → [0.10, -0.01, 0.85, ...]
"real-time message passing"     → [0.11, -0.02, 0.83, ...]   close, even with no shared words
```

### BM25 (keywords)

The chunk becomes a list of **words with weights**. Only words that are in the text get a weight, everything else is 0.
That's why it's called a **sparse** vector: mostly zeros.

```
chunk: "Skills - Core Technical Skills: System Design, Microservices, ..., Caching, Message Queues"

sparse vector (simplified, real ones use word ids):
  {"skill": 0.8, "core": 0.6, "technic": 0.6, "system": 0.6, "design": 0.6,
   "microservic": 0.6, "cach": 0.6, "messag": 0.6, "queue": 0.6}
```

The words are made **lowercase**, **stop words are dropped** ("the", "with", "or"), and they are **stemmed**
("queues" → "queue", "messages" → "messag"), so small word changes still match.

**Scoring a question:**

```
question: "Who has worked with RabbitMQ or message queues?"
words:    rabbitmq, messag, queue        ("who", "has", "with", "or" are stop words)

chunk "Core Technical Skills ... Message Queues"   → has messag + queue         → good score
chunk "Experience: Edge ... async messaging (RabbitMQ)" → has rabbitmq + messag → good score
chunk "Education: Bachelor of Technology EEE ..."  → none of the words          → 0, not returned
```

**Rare words count more.** That part of BM25 is called **IDF** (inverse document frequency):

```
"rabbitmq" appears in 4 of 70 chunks   → rare   → high weight
"built"    appears in 40 of 70 chunks  → common → low weight
```

So a match on "RabbitMQ" or "RAGAS" counts much more than a match on "built" or "system".

---

## 4. Libraries

| Library | Used for | Where it runs |
|---|---|---|
| `qdrant-client` (already installed) | store both vectors, run both searches, RRF merge | Qdrant server |
| **`fastembed`** (new, made by Qdrant) | turn text into the BM25 sparse vector, model `Qdrant/bm25` | locally in our app |
| `huggingface_hub` (already installed) | dense vectors, `BAAI/bge-base-en-v1.5` | HF Inference |

**Why `fastembed` + `Qdrant/bm25`:**
- **Small and local.** `Qdrant/bm25` isn't a neural network. It's a tokenizer, a stop-word list and a stemmer. It's fast, free, and doesn't need torch.
- **Built to work with Qdrant's sparse vectors and IDF.**

**Why not the alternatives:**

| Alternative | Why not |
|---|---|
| `rank_bm25` (pure Python) | keeps the whole index in app memory, has to be rebuilt on every upload, can't filter by user, gone on restart |
| Elasticsearch / OpenSearch | a second database to run and keep in sync with Qdrant |
| Postgres full-text search | then we'd merge results from two databases in Python ourselves |
| a learned sparse model (SPLADE) | better quality, but a heavy neural model. BM25 first, and only if the eval says it isn't enough |

Install: `uv add fastembed`. The first use downloads the small `Qdrant/bm25` files to the local cache, once.

---

## 5. What Qdrant does

1. **Stores both vectors in one point.** Each chunk is still one point with one id and one payload, but with
   **two named vectors**: `"dense"` and `"bm25"`.
2. **Keeps a keyword index.** For sparse vectors Qdrant builds an *inverted index* (word → the points that contain it),
   so a keyword search only looks at chunks that contain at least one of the query words.
3. **Works out IDF itself** (`modifier=IDF`). Qdrant counts how many chunks contain each word **across the whole
   collection** and keeps those counts up to date when chunks are added or deleted. We never compute corpus statistics in our code.
4. **Applies our filters** (`uploaded_by`, `document_hash`) in both searches.
5. **Merges both result lists with RRF** and returns the final top k, all in **one request**.

---

## 6. Indexing changes

### 6.1 Make the sparse vectors: `app/embeddings.py`

```python
import asyncio
from functools import cache

from fastembed import SparseTextEmbedding
from qdrant_client import models

SPARSE_MODEL = "Qdrant/bm25"   # in code, not .env: changing it needs a rebuild migration


@cache
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(SPARSE_MODEL)       # loaded once


def to_qdrant(e) -> models.SparseVector:
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


async def sparse_embed_texts(texts: list[str]) -> list[models.SparseVector]:
    """BM25 vectors for chunks (word counts + chunk length)."""
    embs = await asyncio.to_thread(lambda: list(get_sparse_model().embed(texts)))
    return [to_qdrant(e) for e in embs]


async def sparse_embed_query(query: str) -> models.SparseVector:
    """BM25 vector for a question (each word weight 1; Qdrant adds IDF)."""
    [e] = await asyncio.to_thread(lambda: list(get_sparse_model().query_embed(query)))
    return to_qdrant(e)
```

`embed` (for chunks) and `query_embed` (for questions) are different on purpose. The chunk side carries the BM25
term-frequency and length part, and the question side just lists the words.

The bge query prefix (`"Represent this sentence…"`) is **only for dense**. BM25 gets the plain question.
Otherwise words like "represent" and "passages" would become keywords.

### 6.2 Store both vectors: `app/services/indexing_service.py`

```python
dense = await embed_texts([c.text for c in chunks])
sparse = await sparse_embed_texts([c.text for c in chunks])

points = [
    models.PointStruct(
        id=point_id(doc.id, c.metadata.chunk_index),
        vector={"dense": d, "bm25": s},          # two named vectors, one point
        payload={...same as now...},
    )
    for c, d, s in zip(chunks, dense, sparse)
]
```

And:
- `INDEX_VERSION = 4`
- `check_vector_db()` reads the size of the **named** vector: `config.params.vectors["dense"].size`.
  It also checks that the `bm25` sparse vector exists, otherwise the app refuses to start.

### 6.3 New collection: `vector_migrations/0005_hybrid_dense_and_bm25.py`

A collection's vector layout can't be changed in place. Going from one unnamed vector to two named vectors means a
**rebuild**, in the same shape as `0002` and `0004`:

```python
NEW = "resume_chunks_v4"

await client.create_collection(
    NEW,
    vectors_config={"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)},
    sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)
# same payload indexes as v3: uploaded_by, document_id, document_hash
# re-index every active document into NEW, then move the alias v3 -> v4 (v3 kept for rollback)
```

`modifier=models.Modifier.IDF` turns on point 3 of section 5: Qdrant works out IDF itself.

---

## 7. Search changes: `app/services/retriever_service.py`

```python
CANDIDATES = 20            # each search sends its top 20 to the merge; then RRF picks the final top_k
RRF_WEIGHTS = [2.0, 1.0]   # [dense, bm25]: dense counts twice as much (see section 8)


async def search_chunks(user_id, query, top_k=5, document_hashes=None, mode="hybrid"):
    flt = models.Filter(must=[...uploaded_by, optional document_hash..., same as now])

    if mode == "dense":
        response = await qdrant.query_points(
            settings.qdrant_collection, query=await embed_query(query), using="dense",
            query_filter=flt, limit=top_k, with_payload=True)

    elif mode == "sparse":
        response = await qdrant.query_points(
            settings.qdrant_collection, query=await sparse_embed_query(query), using="bm25",
            query_filter=flt, limit=top_k, with_payload=True)

    else:  # hybrid
        response = await qdrant.query_points(
            settings.qdrant_collection,
            prefetch=[
                models.Prefetch(query=await embed_query(query), using="dense", filter=flt, limit=CANDIDATES),
                models.Prefetch(query=await sparse_embed_query(query), using="bm25", filter=flt, limit=CANDIDATES),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=RRF_WEIGHTS)),   # Qdrant merges the two lists
            limit=top_k,
            with_payload=True,
        )
    return [...same result dicts as now...]
```

- **`prefetch`** = "run these searches first". Each one returns its own top 20.
- **`RrfQuery(Rrf(weights=[2, 1]))`** = "merge those lists by rank, and count dense twice as much as BM25".
  (Plain `FusionQuery(fusion=Fusion.RRF)` counts both equally. We tried it first; it was worse, see section 8.)
- **The filter goes in each prefetch**, so both searches only see this user's resumes.
- **`mode`** exists so the eval can compare dense vs sparse vs hybrid on the same golden set.
  `hybrid` (weighted) is the default.
  The API can expose it too, with a default of `"hybrid"`.

`app/schemas/retriever.py`: add `mode: Literal["hybrid", "dense", "sparse"] = "hybrid"` to `SearchRequest`.

---

## 8. RRF (Reciprocal Rank Fusion): how the two lists are merged

**The problem:** the two searches give scores on different scales.

```
dense cosine:  0.47, 0.46, 0.44 ...     (between 0 and 1)
BM25:          8.31, 5.02, 2.77 ...     (no upper limit)
```
Adding them would let BM25 dominate. Scaling them is guesswork.

**RRF ignores the scores and only uses the rank** (1st, 2nd, 3rd…). Each chunk gets:

```
RRF score = 1 / (k + rank in dense)  +  1 / (k + rank in BM25)
```
`k` is a constant. 60 is the common choice, and Qdrant uses its own fixed one. A chunk that isn't in a list adds nothing for that list.

**Example: q04 "Who has worked with RabbitMQ or message queues?"** (k = 60, made-up ranks)

| Chunk | Dense rank | BM25 rank | RRF score |
|---|---|---|---|
| A · Edge (async messaging RabbitMQ) | 1 | 1 | 1/61 + 1/61 = **0.0328** |
| A · Domain Knowledge (RabbitMQ-based) | 2 | 3 | 1/62 + 1/63 = **0.0320** |
| A · Skills - Core Technical (Message Queues) | — (not in top 20) | 2 | 0 + 1/62 = **0.0161** |
| A · Infinite Computer QC project | 3 | — | 1/63 + 0 = **0.0159** |
| A · Verticurl project | 4 | — | 1/64 + 0 = **0.0156** |

- Chunks that **both** searches like go to the top.
- The **Message Queues** skill chunk, which dense missed completely, gets in **through BM25**. That's exactly the miss we want to fix.
- A chunk only one search found can still make it, but behind chunks both found.

**Where it happens:** inside Qdrant, in the same `query_points` call. We write no merging code.
Just to show the idea, it's this simple:

```python
def rrf(dense_ids, sparse_ids, weights=(2.0, 1.0), k=60):
    scores = {}
    for ranked, weight in zip((dense_ids, sparse_ids), weights):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0) + weight / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

### Weighted RRF (what we use)

Plain RRF counts both lists **equally**. Weighted RRF multiplies each list's points by a weight:

```
score = 2 × 1/(k + rank in dense)  +  1 × 1/(k + rank in BM25)        RRF_WEIGHTS = [2.0, 1.0]
```

**Why:** with equal weights, BM25 matching one common word could push a wrong chunk above the right one.

**Example: q18 "Which candidates have an electrical engineering degree?"** (k = 60, made-up ranks).
The right chunk says "EEE", not "electrical engineering", so BM25 can't find it.

| Chunk | Dense rank | BM25 rank | Plain RRF | Weighted RRF (2 : 1) |
|---|---|---|---|---|
| ✅ Education (EEE) | 1 | — | 1/61 = 0.0164 | 2/61 = **0.0328** ← #1 |
| ❌ Project with the word "engineering" | 6 | 1 | 1/66 + 1/61 = **0.0316** ← #1 | 2/66 + 1/61 = 0.0303 |

- **Plain:** BM25 alone lifts the wrong chunk to the top.
- **Weighted:** dense is sure about Education, and BM25 can't overrule it.

BM25 still rescues exact words. In q04, the "Message Queues" skill chunk is about #8 in dense and #1 in BM25,
which is still enough to get into the top 5.

**In short: dense decides, BM25 breaks ties and rescues exact keywords.**

| mode | recall@5 | precision@5 | MRR |
|---|---|---|---|
| dense | 0.94 | 0.50 | 0.97 |
| hybrid, plain RRF | 0.93 | 0.50 | 0.93 |
| **hybrid, weighted RRF 2 : 1** | **0.95** | **0.51** | 0.95 |

The weights live in `RRF_WEIGHTS` in `app/services/retriever_service.py`. If you change them, run the eval again.

---

## 9. What changes about scores

With hybrid, `score` in the results is the **RRF score** (e.g. `0.0328`), not cosine similarity:

- It only says "how well ranked across both searches". **0.03 doesn't mean bad.**
- It **can't be used for the no-answer check.** RRF always gives the best chunk a high fused score, even for "who knows COBOL?".
- So the no-answer metric in the eval keeps using the **dense** score (a separate `mode="dense"` call for those questions).
  A reranker (bge-reranker-base) was tried for this on 2026-11-18: its scores did not separate answers from no-answers, so it was removed.

---

## 10. Eval plan (how we decide it's better)

`evals/retrieval_eval.py` gets `--mode dense|sparse|hybrid`, and the report records the mode:

```bash
uv run python evals/retrieval_eval.py --mode dense     # same as today's search
uv run python evals/retrieval_eval.py --mode sparse    # keywords only
uv run python evals/retrieval_eval.py --mode hybrid    # both + RRF
```

Report file: `outputs/evals/<date>__bge-base-en-v1.5__v4__k5__hybrid.json`

**Add keyword questions to the golden set** (golden version 3), where exact names matter:

```json
{"id": "q25", "type": "keyword", "question": "SpiceDB experience",
 "expected": [{"document": "resume_a", "contains": "SpiceDB authorization schemas"}]}
{"id": "q26", "type": "keyword", "question": "Who used LightFM?",
 "expected": [{"document": "resume_a", "contains": "embeddings / LightFM"},
              {"document": "resume_b", "contains": "embeddings / LightFM"},
              {"document": "resume_b", "contains": "AI/ML: LightFM"}]}
{"id": "q27", "type": "keyword", "question": "ZMQ to RabbitMQ migration",
 "expected": [{"document": "resume_a", "contains": "PHP + ZMQ"},
              {"document": "resume_b", "contains": "PHP + ZMQ"},
              {"document": "resume_a", "contains": "legacy ZMQ messaging"},
              {"document": "resume_b", "contains": "legacy ZMQ messaging"}]}
```
(Snippets must be checked against the v4 chunks before use: each must match exactly one chunk.)

**Ship rule:**
- hybrid recall@5 **> dense recall@5**, and q04 / q10 improve
- **no question type gets worse**, especially `paraphrase`, where dense is strong and BM25 is weak
- if sparse alone beats hybrid on some types, write it down. It may mean `CANDIDATES` is too small.

---

## 11. Files that change

| File | Change |
|---|---|
| `pyproject.toml` | `uv add fastembed` |
| `app/embeddings.py` | `SPARSE_MODEL`, `sparse_embed_texts()`, `sparse_embed_query()` |
| `app/services/indexing_service.py` | store `{"dense": …, "bm25": …}`; `INDEX_VERSION = 4`; startup check for named vectors |
| `vector_migrations/0005_hybrid_dense_and_bm25.py` | new: build `resume_chunks_v4` with dense + bm25 (IDF), re-index, move alias |
| `app/services/retriever_service.py` | `mode` = dense / sparse / hybrid; hybrid = 2 prefetches + RRF |
| `app/schemas/retriever.py` | `mode` field in `SearchRequest` |
| `evals/retrieval_eval.py` | `--mode` flag; no-answer check uses dense scores |
| `evals/retrieval_golden.json` | version 3: keyword questions |
| `tests/conftest.py` | fake sparse embedder (no model download in tests) |
| `tests/test_retriever.py` | each mode respects the user / document filters; hybrid returns chunks found by either search |
| `docs/vector_migrations.md` | current state → v4, `0005` |

---

## 12. Risks and limits

- **Tech words and tokenization.** BM25 splits on punctuation, so `C++` → `c`, `CI/CD` → `ci` + `cd`, `Node.js` → `node` + `js`.
  Mostly fine, but very short tokens like `c` match a lot. Watch for it in the eval.
- **Typos aren't matched.** `Kubernets` ≠ `Kubernetes` for BM25. Dense still helps here, which is one more reason to keep both.
- **Our duplicate resumes.** Resume A and B share most text, so BM25 often ranks both copies together. Same as dense today.
- **Chunk length.** BM25 favours short chunks with the word in them. Our chunks are now small and focused, which suits it.
  `Qdrant/bm25` has an average-length setting (`avg_len`) we could tune if the eval asks for it.
- **The in-memory test Qdrant** (`":memory:"`) must support sparse vectors, IDF and RRF fusion. Check this first.
  If any part isn't supported, test the fusion path against the real Qdrant only, and keep in-memory tests for dense and sparse separately.

---

## 13. Order of work

1. `uv add fastembed`. Try `Qdrant/bm25` on a few chunks and look at the tokens.
2. `app/embeddings.py`: sparse functions plus tests.
3. Migration `0005` + indexing change + `INDEX_VERSION = 4` + startup check.
4. `retriever_service.py`: the three modes.
5. Eval `--mode` + keyword questions (golden v3, snippets checked).
6. Run the eval in all three modes, compare with the v3 report, apply the ship rule.
7. Update `docs/vector_migrations.md`.
