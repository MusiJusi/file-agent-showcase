# Contributing

Use Python 3.12 and the checked-in lockfile:

```bash
uv sync --python 3.12 --frozen
```

## Workflow

Create a focused branch from `main`, make a small change, and open a pull request
with the motivation and verification steps. Issues are useful for larger changes
but are not required for documentation fixes.

Prefer descriptive commits such as `docs: clarify local setup` or
`refactor: group table tools`. CI should pass before merging.

## Architecture

Read [the architecture map](docs/architecture.md) before changing layer boundaries.
Keep parsing, retrieval, model adapters, and answering separate. Preserve source
metadata through every tool result used as answer evidence.

- Add tools to `src/file_agent/agent_tools/` and register them in `DOCUMENT_TOOLS`.
- Put generation adapters in `src/file_agent/benchmarks/`.
- Keep provider-specific settings in the LLM/VLM factories and environment.
- Preserve the documented CLI commands and legacy import aliases.
- Add behavior tests under the corresponding `tests/` subsystem.
- Use fake LLM/VLM clients in tests; do not call real model endpoints.

## Checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .

cd eval_pipeline
uv sync --python 3.12 --frozen
uv run pytest tests
uv run ruff check .
uv run ruff format --check .
```

The evaluator has its own dependency environment. Run its tests from
`eval_pipeline/`, not as part of the application's pytest invocation.

Update examples and documentation when changing configuration or commands.
Keep credentials, source documents, model caches, generated answers, and raw judge
traces out of Git. Use synthetic documents and placeholder credentials in examples.
