# Project background

File Agent grew from a team project for document question answering.
This is a separate portfolio edition with a curated layout.

**MusiJusi was responsible for the RAG logic**, including retrieval
orchestration, context assembly, grounded answer generation, and the LangGraph
agent with its evidence-aware document tools.

## Team foundation

The shared codebase includes the Document/Block contract, format-aware parsers,
OCR and visual enrichment, structural chunking, hybrid retrieval, model adapters,
and the initial Streamlit application. The detailed
[parsing/chunking](parsing_and_chunking.md) and [retrieval](retrieval.md)
research notes preserve that development context.

## RAG ownership and contribution

MusiJusi's responsibility was the logic that connects document retrieval to
answers supported by source evidence:

| Area | Modules |
| --- | --- |
| End-to-end RAG orchestration and context assembly | `rag.py`, `rag_core.py`, `qa.py` |
| LangGraph QA and tool-agent execution | `rag_graph.py` |
| Evidence-aware document tools and source propagation | `agent_tools/` |

This edition also brings together deterministic table operations, compact
conversation state, Hugging Face and DocBench generation workflows, and
resumable judge evaluation. Its presentation cleanup groups components by
responsibility while preserving public imports and commands.

## Golden set and published evaluation

The project's golden set contains **100 files and 400 questions**.
The published archived result covers **125 selected questions** from an original
127-question experiment. It is a subset evaluation, not a result over all 400
golden-set questions. See [evaluation](evaluation.md) for the scores and protocol.

## Attribution

The underlying project is collaborative work, with **MusiJusi responsible for
the RAG logic**. Other modules, experiments, and design decisions include team
contributions. The original development history is retained in a separate
private archive.
