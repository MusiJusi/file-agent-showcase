# Hugging Face generation

The runner loads QA rows, downloads their referenced documents, generates answers,
and exports the exact evidence, tool calls, token counts, and timings used.
Implementation: `src/file_agent/benchmarks/hf/`.

## Prepare

Configure an answering endpoint using [the setup guide](local_inference.md).
The dataset must provide `id`, `question`, `answer`, and `doc_ids` columns.
References are retained for evaluation; they are not passed to the answering
model as ground truth. Use `--revision` to pin the dataset commit.

Run from the repository root. Choose a fresh output directory for each
configuration. Output directories under `runs/` are ignored by Git.

## Pilot

```bash
uv run python generate_hf_dataset.py \
  --dataset-id sandrik1271/RAG-QA-Dataset \
  --output-dir runs/hf-pilot \
  --env-file .env \
  --rag-mode tool_agent \
  --max-tool-rounds 8 \
  --top-k 5 \
  --max-tokens 2000 \
  --limit 3
```

In PowerShell, the same command can be entered on one line:

```powershell
uv run python generate_hf_dataset.py --dataset-id sandrik1271/RAG-QA-Dataset --output-dir runs/hf-pilot --env-file .env --rag-mode tool_agent --max-tool-rounds 8 --top-k 5 --max-tokens 2000 --limit 3
```

Inspect answers, sources, and failures before scaling up.

## Full run and resume

Remove `--limit 3` and use a new directory such as `runs/hf-full`.
If interrupted, repeat the **same command**, selection, configuration, and output
directory with `--resume`. Successful checkpoints are validated and reused;
changed parameters cannot silently mix in one run.

Use repeatable `--record-id` options for exact row selections. Do not combine
them with `--limit`. `--continue-on-error` records failures and proceeds; these
diagnostic rows need review before evaluation. `--record-timeout` is supported
on Linux/macOS, as described by the CLI.

```bash
uv run python generate_hf_dataset.py --help
```

The CLI accepts the historical `--max-chars` / `--overlap` settings for
compatibility. Structured chunking is principally governed by the encoder's
token window and `CHUNK_TARGET_TOKENS`; do not infer embedded token length from
the character setting alone.

## Artifacts

| Path | Content |
| --- | --- |
| `answers.parquet` | Answers, references, exact contexts, and diagnostics |
| `hf_dataset/` | Equivalent Hugging Face dataset for `load_from_disk` |
| `run_manifest.json` | Dataset revision, generation settings, result counts |
| `checkpoints/` | Per-row checkpoints for resumable generation |

Keep raw results and source documents outside the committed repository.
See [evaluation](evaluation.md) for judging the generated Parquet file.
