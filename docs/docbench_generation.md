# DocBench generation

The runner processes a local [DocBench checkout](https://github.com/Anni-Zou/DocBench).
It discovers questions from numeric document folders:

```text
DocBench/data/
  0/
    <document>.pdf
    0_qa.jsonl
  1/
    <document>.pdf
    1_qa.jsonl
```

Implementation: `src/file_agent/benchmarks/docbench/`.

## Execution policy

The runner parses and indexes each PDF once for all selected questions in that
folder. Questions are stateless. Reference answers, evidence annotations,
question types, and domains are copied to outputs for evaluation, not sent to
the answering agent.

Successful answers are checkpointed atomically. Failed questions are recorded
and processing continues by default; `--fail-fast` changes that policy.
Resume validates model, prompt, parsing/retrieval configuration, source PDF, and
question fields before reusing a checkpoint.

## Pilot

Configure [the model endpoint](local_inference.md), then run from the repository
root. Replace `data/DocBench/data` with your own dataset directory:

```bash
uv run python generate_docbench.py \
  --data-dir data/DocBench/data \
  --output-dir runs/docbench-pilot \
  --env-file .env \
  --rag-mode tool_agent \
  --max-tool-rounds 8 \
  --top-k 5 \
  --max-tokens 2000 \
  --folder-id 0 \
  --limit 3
```

PowerShell equivalent:

```powershell
uv run python generate_docbench.py --data-dir data/DocBench/data --output-dir runs/docbench-pilot --env-file .env --rag-mode tool_agent --max-tool-rounds 8 --top-k 5 --max-tokens 2000 --folder-id 0 --limit 3
```

## Full run and resume

After inspecting the pilot, remove `--folder-id 0` and `--limit 3`, and choose
a new directory such as `runs/docbench-full`.

To continue after an interruption, repeat the same command with `--resume`.
Use the original output directory and settings. A changed configuration is
rejected to prevent mixing incomparable answers.

Subset selectors include `--folder-start`, `--folder-end`, repeatable
`--folder-id`, `--record-id`, `--domain`, `--question-type`, and `--limit`.

```bash
uv run python generate_docbench.py --help
```

## Outputs

| Path | Content |
| --- | --- |
| `answers.parquet`, `hf_dataset/` | Answers and diagnostics |
| `predictions.jsonl` | Lossless generation records |
| `docbench_eval_input.jsonl` | DocBench judge-compatible records |
| `summary.json` | Latency, usage, and failure statistics |
| `run_manifest.json` | Selection and generation parameters |
| `run_config.json` | Resume contract |
| `checkpoints/` | Per-question checkpoints |
| `generation.log` | Persistent generation log |

Inspect failure counts and the selected question count rather than assuming
every dataset checkout has the same release size. Keep these artifacts and the
dataset out of Git; `data/` and `runs/` are ignored.
