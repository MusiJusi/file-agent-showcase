# File Agent Pipeline

RAG pipeline for answering questions about documents:

- parse PDF, DOCX, Markdown, TXT, HTML, XLSX, and PPTX;
- split documents into chunks;
- retrieve relevant chunks with LanceDB;
- generate an answer with an OpenAI-compatible LLM.

Every format is parsed into typed blocks (headings with levels, paragraphs,
lists, Markdown tables, figures, code): PDF via Docling, DOCX via python-docx,
PPTX/XLSX/HTML/Markdown/TXT with dedicated structure-aware parsers. Chunks are
section-coherent, prefixed with their heading path and budgeted in the
retrieval encoder's tokens (Qwen3-Embedding-0.6B by default). Scanned PDF pages are detected
per page and transcribed by the multimodal chat model (Qwen3.5); figures in
PDF/DOCX/PPTX are described by the same model. Design and evaluation results:
[docs/parsing_and_chunking.md](docs/parsing_and_chunking.md).

Retrieval is hybrid (BM25 over a lemmatised Russian view + Qwen3 dense,
reciprocal-rank fusion), optionally widened with model-written formulations of
the question (multi-query / HyDE), re-scored by Qwen3-Reranker-0.6B and cut to
distinct passages. Stages, switches, the retrieval-level benchmark and an audit
of the judge: [docs/retrieval.md](docs/retrieval.md).

## Agent layer

The same parsing, chunking, and retrieval pipeline can be used in two modes:

- `RAG_MODE=standard` performs one retrieval pass and generates an answer;
- `RAG_MODE=tool_agent` runs a bounded LangGraph loop with native LLM tool calls.

The tool agent can search and open passages, inspect document metadata and
outlines, count literal occurrences over a complete document, read a whole
document or a specific section/page/slide/sheet, inspect and aggregate tables,
compare columns across tables, perform deterministic arithmetic, and analyze a
PDF page or figure with the configured VLM. Tool observations are returned as
normal `RAGResponse.sources`, so the UI and dataset exporters preserve the
evidence used for the answer.

For evaluation runs that require document evidence, the agent now routes the
first call by question type: metadata/count, table, visual, location, or semantic
search. Table row lookups and averages are deterministic, metadata reports both
raw and content-level visual counts, and multi-document turns require independent
evidence from every named file. Long tool traces are compacted before they reach
the model while full source passages remain available in `RAGResponse.sources`.

The Streamlit application provides multiple persistent chats for the current
set of uploaded documents. Conversation memory keeps only compact user/final
answer pairs; tool traces are scoped to the active turn. Configure the loop with
`RAG_MAX_TOOL_ROUNDS`, `RAG_HISTORY_TURNS`, and `RAG_MAX_CONTEXT_TOKENS` in `.env`.
The context budget defaults to 90,000 estimated input tokens, leaving headroom for
the answer on a 131k model. The selected LLM endpoint must support OpenAI-compatible
native tool calling in agent mode.

The Hugging Face generator supports reproducible `standard`/`tool_agent` runs
with per-row checkpoints and exported tool/token/timing diagnostics. The local
DocBench runner parses and indexes each PDF once for all of its questions,
continues past isolated failures, and writes judge-ready JSONL plus
Parquet/Hugging Face artifacts. See
[docs/hf_dataset_generation.md](docs/hf_dataset_generation.md) and
[docs/docbench_generation.md](docs/docbench_generation.md).

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Configure the LLM backend in `.env`.

To run the agent instead of the standard RAG path:

```env
RAG_MODE=tool_agent
RAG_MAX_TOOL_ROUNDS=4
RAG_HISTORY_TURNS=6
RAG_MAX_CONTEXT_TOKENS=90000
```

## Run the application

```bash
uv run streamlit run app.py
```

Local inference setup: [docs/local_inference.md](docs/local_inference.md).

## Tracing (OpenTelemetry, Jaeger, and Langfuse)

The pipeline is instrumented with OpenTelemetry (`src/file_agent/telemetry.py`).
Spans are created in `parse_file`, `chunk_document`, `load_documents`,
`index_documents`, `LanceDBRetriever.index`/`.search`, `answer_question_with_context`
and `answer_indexed_documents`, and exported over OTLP/gRPC to Jaeger.

Run Jaeger locally:

```bash
docker compose up -d jaeger
```

The UI is available at http://localhost:16686 (service `file-agent-pipeline`).

Spans are sent to `localhost:4317` by default. Override the endpoint with the
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable (see `.env.example`).

Langfuse export is optional. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_BASE_URL` together to send the same trace tree through Langfuse's
OTLP/HTTP endpoint. Leaving all three blank disables that exporter.

If Jaeger isn't running, `configure_telemetry()` still works — spans are sent
in the background via `BatchSpanProcessor` and simply won't arrive anywhere.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
