# RAGAS evaluation

Status: steps 1-2 **done** (spike, phase 1 code); step 3 (full run + hand check) next.

## 1. Why

Our evals today measure **whether the system found the right thing**:

| Eval | Measures | Exact? |
|---|---|---|
| `retrieval_eval.py` | recall@5, precision, MRR against golden chunks | yes |
| `planner_eval.py` | route / sections right | yes |
| `answer_eval.py` | answer rate, a cited source is a golden chunk, "not found" right, list items named | yes |

None of them checks **whether the generated text is true and on topic**. Code already checks that every `[n]` points
at a real chunk, but not that the chunk *says* what the answer claims. That gap is real: the match verdict once said a
candidate "built APIs for payment processing" while "payment" was in that resume 0 times (`jd_matching_plan.md`).

RAGAS closes it with an LLM **judge** that reads the answer and the chunks and scores them.

## 2. Where RAGAS is used (and where not)

| Part | RAGAS | Metric | Why |
|---|---|---|---|
| **Q&A answer** (`/qa/ask`) | ✅ phase 1 | faithfulness, answer relevancy | classic RAG: question → chunks → generated answer |
| **Match verdict** (strengths) | ✅ phase 2 | faithfulness | generated claims with cited chunks; where a hallucination was already seen |
| **Q&A retrieval quality** | ✅ phase 3 | context precision, context recall | needs a short reference answer per question (we have only expected chunks today) |
| search, planner, JD / resume parsers, skill matching, score | ❌ | – | already measured **exactly** (code or golden labels); an LLM judge would be slower, cost tokens, and be less exact |
| "not found" questions (9) | ❌ | – | RAGAS has no good metric for "the resumes don't say"; our own no-answer accuracy stays |

## 3. The metrics, in plain words

| Metric | Question | How the judge works | Needs |
|---|---|---|---|
| **Faithfulness** | Is every claim in the answer supported by the chunks? | splits the answer into claims, checks each one against the chunks; score = supported claims / all claims | question, answer, chunks |
| **Answer relevancy** | Does the answer address the question? | writes questions the answer would fit, compares them with the real question (embeddings) | question, answer, embeddings |
| **Context precision** | Are the useful chunks ranked first? | marks each chunk useful / not for the reference answer, rewards useful ones at the top | question, chunks, reference |
| **Context recall** | Do the chunks hold everything the reference needs? | splits the reference into claims, checks each one is in the chunks | chunks, reference |

Example, faithfulness:

```
Q:       Who has worked with RabbitMQ?
chunks:  [1] "...Moved order events to RabbitMQ consumers"   [2] "...Skills: Python, SQL"
answer:  "Asha used RabbitMQ for order events [1] and Kafka for payments [2]."
claims:  "Asha used RabbitMQ for order events"  → supported by [1]   ✅
         "Asha used Kafka for payments"          → not in any chunk   ❌
faithfulness = 1 / 2 = 0.5
```

## 4. The judge model

| Candidate | Served via the HF router (checked 2026-11-25) | Verdict |
|---|---|---|
| `Qwen/Qwen2.5-72B-Instruct` | deepinfra | ❌ **not as judge**: it wrote the answers; a model grading itself is too lenient |
| **`openai/gpt-oss-120b`** | **cerebras 1.0 s, groq 1.7 s, deepinfra 2.8 s** | ✅ **judge**: strong, another model family, 3 providers |
| `meta-llama/Llama-3.3-70B-Instruct` | deepinfra 0.9 s | backup judge |
| `meta-llama/Llama-3.1-8B-Instruct` | groq, deepinfra | ❌ too weak to judge reliably |

Settings (same pattern as the agents, same router URL and `HF_TOKEN`; but **not** `build_hf_chat`: RAGAS 0.4 metrics
want an *instructor* LLM made by `ragas.llms.llm_factory(model, client=AsyncOpenAI(base_url=ROUTER_URL, ...))`, see
section 12):

```python
judge_model_id: str = "openai/gpt-oss-120b"
judge_provider: str = "cerebras"
judge_fallback_provider: str = "groq"
judge_max_tokens: int = 4096     # gpt-oss reasons first; a small limit gives an empty reply (seen in the router check)
judge_timeout: int = 120
```

Embeddings for answer relevancy: our own bge (`app/embeddings.py`), through a small adapter class: a subclass of
RAGAS's `BaseRagasEmbedding` (`embed_text` / `aembed_text` / `aembed_texts`) that calls our `embed_texts` (no bge query
prefix: it compares a question with questions, so both sides are embedded the same way).

**Trusting the judge**: a judge can be wrong too. Phase 1 ends with a hand check of ~10 judged answers (do I agree
with each "supported / not supported"?). If agreement is poor, change the judge before trusting any number.

## 5. Data flow

### Phase 1: Q&A

```
evals/answer_eval.py  (already runs the golden questions through answer_question, planner on / off)
   │  per answerable question: question, answer, the chunks the answer LLM SAW, cited sources
   ▼
evals/ragas_eval.py   (new)  RAGAS samples: user_input = question, response = answer, retrieved_contexts = chunks
   │  judge = gpt-oss-120b, embeddings = bge
   ▼
report: faithfulness + answer relevancy per question, averages per question type and per mode (planner on / off)
saved to outputs/evals/<date>__ragas__<mode>.json
```

**One code change in the app is needed**: `answer_question` returns only the *cited* sources, but faithfulness must
be judged against **every chunk the answer LLM saw** (a claim can come from an uncited chunk). It will also return
the chunks it sent, under an internal key `chunks`. `AskResponse` (the API schema) doesn't list it, so FastAPI drops
it: the API response doesn't change.

Which questions: the 25 answerable golden questions + the 12 list questions = **37**. The 9 "not found" questions are
skipped (section 2).

### Phase 2: match verdicts

```
per (JD, top candidate):  user_input = "How well does <candidate> fit the <JD title> job?"
                          response   = the strengths, one per line ([n] markers removed)
                          retrieved_contexts = the evidence chunks given to the MatchAgent (build_evidence)
→ faithfulness per verdict
```

Gaps are **not** scored by faithfulness (a gap is a claim that something is *absent*, which a "supported by the chunks"
check can't judge). Instead, code checks each gap mentions a skill from the code's `missing` lists or the years.

Data: the sample JDs (payments, LLM engineer, data engineer) × the top 3 candidates = ~9 verdicts. Small: this mainly
proves the strengths are grounded. A bigger set comes with the synthetic resumes (interview plan, item 1).

### Phase 3: context precision / recall

Add a one-line `reference` answer to each answerable golden question (`evals/retrieval_golden.json`), e.g.
q04 "Who has worked with RabbitMQ?" → "Yogesh: migrated PHP + ZMQ to FastAPI + RabbitMQ; RabbitMQ in skills".
Then add the two context metrics. ~25 lines to write by hand.

## 6. What the report looks like

```
RAGAS eval  2026-11-xx  judge openai/gpt-oss-120b @ cerebras   (37 questions, planner on)

                 n   faithfulness   answer_relevancy
overall         37          0.93              0.88
skill            6          0.97              0.91
list            12          0.90              0.84
...

lowest faithfulness
  q04  0.50  "Who has worked with RabbitMQ?"  unsupported: "used Kafka for payments"
```

Every unsupported claim is printed (RAGAS keeps the judge's per-claim verdicts), so a low score points at the exact
sentence to look at.

**What we do with it**: faithfulness < 0.9 on a question → read it; a real hallucination → fix the prompt / chunking
and re-run. Answer relevancy is a softer signal (short answers like "Yogesh [1], [2]" score lower: this is how we
decide the "short answers" item in `todo/answer-step.md` with a number, not by feel).

## 7. Cost and time

| | per run |
|---|---|
| answers (existing answer eval, planner on) | 37 × (planner + Qwen) |
| faithfulness | ~2 judge calls per answer (claims, then verdicts) → ~75 |
| answer relevancy | ~1 judge call + embeddings per answer → ~37 |
| total judge calls | **~110**, a few minutes with 3 in parallel |

Run it by hand after a prompt / retrieval change, not on every commit.

## 8. Safety and privacy

- The judge gets resume chunks, so resume text goes to a **second provider** (cerebras / groq, besides deepinfra).
  Fine for the golden resumes (yours, for development); for real users' data an eval would need consent or synthetic
  resumes (the interview plan's synthetic set fixes this too).
- No resume text in the saved report except the question, the answer and the judge's unsupported claims (needed to
  act on a low score). Reports stay in `outputs/` (git-ignored).
- Keys only from `.env` (`HF_TOKEN`); RAGAS's own telemetry turned off (`RAGAS_DO_NOT_TRACK=true`).

## 9. Code changes

| File | Change |
|---|---|
| `pyproject.toml` | `ragas==0.4.4` as a **dev** dependency (pinned: its API changed between 0.1, 0.2 and 0.4) |
| `app/core/config.py`, `.env.example` | `judge_*` settings |
| `app/services/qa_service.py` | result also carries `chunks` (internal, not in the API response) |
| `evals/ragas_adapters.py` | **new**: judge LLM (from `build_hf_chat`) and bge embeddings wrapped for RAGAS |
| `evals/ragas_eval.py` | **new**: phase 1 (`--part qa`, `--mode planner/plain`), phase 2 (`--part match`) |
| `evals/answer_eval.py` | share the question-building / answering code with `ragas_eval.py` instead of copying it |
| `tests/test_ragas_eval.py` | **new**, no network: samples built right (all chunks, not only cited; "not found" skipped; `[n]` removed for match); the adapter returns bge vectors; report math |
| `evals/retrieval_golden.json` | phase 3: `reference` per answerable question |

## 10. Order of work

1. ✅ **Spike**: `ragas==0.4.4` installed (dev), 3 made-up samples judged on cerebras and groq. Results in section 12.
2. ✅ `chunks` in `answer_question` + adapters + `ragas_eval.py` + tests (7, offline). `qa_agent.format_chunk` is
   shared by the QA agent and the eval, so the judge sees the same chunk text. `--part match` comes with step 4.
3. Run on 37 questions, planner on and off. **Hand-check ~10 judgements** (section 4). Write the results here.
4. Phase 2: verdict faithfulness + the code check for gaps.
5. Phase 3: reference answers, context precision / recall.
6. Later: push the scores to Langfuse traces (observability plan).

## 11. Risks

| Risk | Plan |
|---|---|
| RAGAS API differs in 0.4 from older examples online | the spike (step 1) uses the installed version's docs; version pinned |
| gpt-oss is a reasoning model: empty or slow replies with small limits | `judge_max_tokens` 4096; spike measures it; backup judge Llama 3.3 70B |
| small sample (2 people, 37 questions) | numbers are a baseline, not a verdict; grows with the synthetic resume set |
| the judge is wrong | hand-check ~10 judgements before trusting the averages |
| one provider rate-limits | judge fallback provider (groq), 3 calls in parallel |

## 12. Spike results (2026-11-27)

3 made-up samples, judge `openai/gpt-oss-120b`, `max_tokens` 4096, temperature 0:

| Sample | Faithfulness | Answer relevancy | Right? |
|---|---|---|---|
| "Asha moved order events to RabbitMQ [1]" (all supported) | 1.00 | 0.76 | ✅ |
| "...RabbitMQ [1] and Kafka for payments [2]" (Kafka made up) | **0.50**, Kafka claim marked unsupported with a reason | 0.53 | ✅ |
| "The resumes don't mention Rust" | 1.00 | 0.76 | ok (skipped in the eval anyway) |

Same verdicts on cerebras (~1.2 s per faithfulness, ~2.7 s per relevancy) and groq (~2 s / ~3.7 s). Structured
output works with the default instructor JSON mode through the router.

What changed in the plan:

1. **The judge must see the chunks exactly as the answer LLM saw them**, header included
   (`[n] candidate: Asha Rao | resume: ab12cd` + text, from `qa_agent.build_context`). With bare chunk text the judge
   marked "Asha Rao moved order events..." **unsupported**: no chunk says "Asha Rao". Correct of the judge, wrong
   input from us. So the samples are built with the same header code as the QA agent.
2. **LLM**: `llm_factory(...)` with an `AsyncOpenAI` client on the router, not `build_hf_chat` (LangChain models are
   rejected by 0.4 metrics). No `BaseAgent` retry: on a failed judge call the question is recorded as an error and
   skipped from the averages (re-run, or `--provider groq`).
3. **Embeddings**: subclass `BaseRagasEmbedding`, not LangChain `Embeddings`.
4. **Unsupported claims**: `Faithfulness.ascore` returns only the number, so the eval calls its two steps
   (`_create_statements`, `_create_verdicts`) and keeps each claim's verdict + reason. Private methods: fine with the
   version pinned; a test notices if an upgrade renames them.
5. **Answer relevancy numbers run low** with bge (a fully on-topic answer: 0.76; bge cosine between paraphrased
   questions is ~0.7-0.9, not 1.0). Read it **relatively** (planner on vs off, before vs after a prompt change), not
   as "0.76 is bad".
