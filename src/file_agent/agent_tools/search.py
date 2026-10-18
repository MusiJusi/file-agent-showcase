"""Hybrid document search and retrieval-context expansion."""

from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.agent_tools.common import _find_document, _tool_command, _truncate_passage
from file_agent.agent_tools.limits import (
    LLM_CONTEXT_METADATA_KEY,
    MAX_TOOL_QUERY_LENGTH,
    MAX_TOOL_TOP_K,
)
from file_agent.chunking import Chunk
from file_agent.qa import select_context_passages
from file_agent.retrieval import SearchResult


@tool
def search_documents(
    query: str,
    runtime: ToolRuntime[Any, dict],
    top_k: int | None = None,
    source_file: str | None = None,
) -> Command:
    """Search uploaded documents for passages relevant to a query.

    Args:
        query: A concise semantic search query.
        top_k: Number of retrieval results to return, from 1 to 10. Uses the
            active RAG top-k setting when omitted.
        source_file: Optional exact file name returned by list_documents.
    """
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if len(normalized_query) > MAX_TOOL_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_TOOL_QUERY_LENGTH} characters")

    resolved_source: str | None = None
    if source_file is not None:
        if not source_file.strip():
            raise ValueError("source_file must not be empty")
        resolved_source = _find_document(runtime.context.documents, source_file).file_name

    requested_top_k = runtime.context.default_top_k if top_k is None else top_k
    bounded_top_k = max(1, min(requested_top_k, MAX_TOOL_TOP_K))
    if resolved_source is None:
        results = runtime.context.retriever.search(
            query=normalized_query,
            top_k=bounded_top_k,
        )
    else:
        results = runtime.context.retriever.search(
            query=normalized_query,
            top_k=bounded_top_k,
            source_file=resolved_source,
        )
    payload = _serialize_search_results(normalized_query, results, resolved_source)
    evidence_sources = _search_evidence_sources(results)
    return _tool_command(
        payload,
        runtime,
        sources=evidence_sources,
        search_query=normalized_query,
    )


@tool
def read_source_context(
    chunk_id: str,
    runtime: ToolRuntime[Any, dict],
) -> Command:
    """Read the stored surrounding passage for a previously found search result.

    Args:
        chunk_id: Exact chunk_id returned by search_documents in this agent run.
    """
    normalized_id = chunk_id.strip()
    if not normalized_id:
        raise ValueError("chunk_id must not be empty")

    result = next(
        (item for item in runtime.state.get("sources", []) if item.chunk.id == normalized_id),
        None,
    )
    if result is None:
        raise ValueError(
            "chunk is not available; call search_documents first and use a returned chunk_id"
        )

    metadata = dict(result.chunk.metadata)
    passage = str(metadata.pop("context", None) or result.chunk.text)
    metadata.pop(LLM_CONTEXT_METADATA_KEY, None)
    payload = {
        "chunk_id": result.chunk.id,
        "text": passage,
        "metadata": metadata,
    }
    updated_metadata = dict(result.chunk.metadata)
    updated_metadata[LLM_CONTEXT_METADATA_KEY] = passage
    updated_source = SearchResult(
        chunk=Chunk(
            id=result.chunk.id,
            text=result.chunk.text,
            metadata=updated_metadata,
        ),
        score=result.score,
    )
    return _tool_command(payload, runtime, sources=[updated_source])


def _serialize_search_results(
    query: str,
    results: list[SearchResult],
    source_file: str | None = None,
) -> dict:
    serialized = []
    for rank, (result, passage) in enumerate(select_context_passages(results), start=1):
        metadata = dict(result.chunk.metadata)
        metadata.pop("context", None)
        serialized.append(
            {
                "rank": rank,
                "chunk_id": result.chunk.id,
                "score": result.score,
                "text": _truncate_passage(passage),
                "metadata": metadata,
            }
        )
    return {
        "query": query,
        "source_file": source_file,
        "results_count": len(serialized),
        "results": serialized,
    }


def _search_evidence_sources(results: list[SearchResult]) -> list[SearchResult]:
    evidence_sources = []
    for result, passage in select_context_passages(results):
        metadata = dict(result.chunk.metadata)
        metadata[LLM_CONTEXT_METADATA_KEY] = _truncate_passage(passage)
        evidence_sources.append(
            SearchResult(
                chunk=Chunk(
                    id=result.chunk.id,
                    text=result.chunk.text,
                    metadata=metadata,
                ),
                score=result.score,
            )
        )
    return evidence_sources
