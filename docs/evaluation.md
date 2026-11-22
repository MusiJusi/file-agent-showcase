# Evaluation

## Golden set

The project's golden set contains **100 files and 400 questions**.

## Archived agent-v6 result

The published means below match the main [README evaluation snapshot](../README.md#evaluation-snapshot).
The scores were produced with **DeepSeek V4 Flash** and **evaluator version 2**:

| Metric | Mean | Interpretation |
| --- | ---: | --- |
| Faithfulness | 0.927 | Support for answer claims in the supplied evidence |
| Answer correctness | 0.634 | Reference-based factual and semantic comparison |
| Answer relevancy | 0.855 | How closely an answer addresses the question |
| Context precision | 0.814 | Ranking quality of useful supplied evidence |
| Context recall | 0.921 | Coverage of reference information by that evidence |

The [sanitized summary](results/agent-v6.json) records these published
means at the same three-decimal precision as the README. Per-question scores
and distribution statistics are not included. It contains no questions,
documents, credentials, cloud folder identifiers, or machine-specific paths.

## Protocol and provenance

- `q0016` and `q0067` were excluded for suspected incorrect reference
  annotations.
- Answers were supplied as a separate generated Parquet artifact, then filtered.
  Its SHA-256 is
  `bc7d9c1d7007319e16f38855d46714b17822717b7339183fd1588816a08c8d11`.
- The judge was **DeepSeek V4 Flash**, through an OpenAI-compatible endpoint.
- The evaluator protocol is **version 2**, using Ragas `AnswerCorrectness`
  with factual/semantic weights 0.75/0.25.
- Judge embeddings used
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  They are separate from Qwen embeddings used for document retrieval.
- The recorded policy allowed 8192 output tokens, a 300-second request timeout,
  a 900-second metric timeout, and three retries. Provider-default reasoning was
  used. Provider-specific fallback behavior is documented in the evaluator.

Earlier generation manifests describe the project setup but do not prove the
exact settings of the separately supplied final answer artifact. No fresh
generation or paid judging was performed during this presentation refactor.

## How to read the scores

These are graded LLM-judge measurements on a 0–1 scale.
A correctness score of 0.634 does **not** mean 63.4% of answers were classified
as correct. Scores depend on the judge, references, evidence, and request policy.

The report's automatic negative-example count was zero. This reflects its
configured detection, not a manual determination that every question was
answerable. Correct abstentions need separate inspection because standard
answer-relevancy scoring can penalize them.

The historical tables in [parsing/chunking](parsing_and_chunking.md) and
[retrieval](retrieval.md) use other answer artifacts, subsets, or judge protocols.
They are engineering context, not a controlled comparison with this evaluation.
Rerun both variants under one protocol before claiming an improvement.

## Reproduce the workflow

1. Connect an endpoint as described in [local inference](local_inference.md).
2. Generate answers with the [HF](hf_dataset_generation.md) or
   [DocBench](docbench_generation.md) runner. Retain manifests and checkpoints.
3. Configure the [independent evaluator](../eval_pipeline/README.md).
4. Run the judge and resume the same output directory after an interruption:

```bash
cd eval_pipeline
uv sync --python 3.12 --frozen
uv run --env-file .env python scripts/run_eval.py \
  --run ../runs/my-run/answers.parquet \
  --out ../evaluations/my-run \
  --batch-size 1 \
  --resume
```

Completed rows are checkpointed after each question. Resume retries incomplete
rows while retaining completed scores. It checks the artifact fingerprint,
judge, embedding model, and policy to prevent silently mixing protocols.

Original answers and raw judge traces are not bundled. Reproducing the exact
archived result requires the same input answers, references, model revision, and
policy. New generation runs can legitimately produce different scores.
