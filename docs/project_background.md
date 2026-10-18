# Project background

File Agent grew from a shared team project for document question answering.
This is a separate portfolio edition with a curated layout.

## Shared foundation

The foundation includes the Document/Block contract, format-aware parsers,
OCR and visual enrichment, structural chunking, hybrid retrieval, model adapters,
and the initial Streamlit application. The detailed
[parsing/chunking](parsing_and_chunking.md) and [retrieval](retrieval.md)
research notes preserve that development context.

## Focus of this edition

This edition brings together the LangGraph agent, source-aware document tools,
deterministic table operations, compact conversation state, Hugging Face and
DocBench generation workflows, and resumable judge evaluation.

Its presentation cleanup groups components by responsibility while preserving
public imports and commands. The archived benchmark is described separately
from the shared baseline's experiments.

## Attribution

The underlying project is collaborative work. Hosting this edition under
MusiJusi does not imply sole authorship of every module, experiment, or design.
The original development history is retained in a separate private archive.
