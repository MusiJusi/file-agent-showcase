import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    LLM_CONTEXT_METADATA_KEY,
    TOOL_AGENT_SYSTEM_PROMPT,
    TOOL_LIMIT_MESSAGE,
    ToolAgentContext,
)
from file_agent.chunking import Chunk
from file_agent.document import Document
from file_agent.llm.base import EmptyLLMResponseError, LLMClient, ToolChoice
from file_agent.qa import answer_question_with_context
from file_agent.rag_core import (
    RAGResponse,
    chunk_documents,
    load_documents,
)
from file_agent.retrieval import Retriever, SearchResult
from file_agent.telemetry import tracer

LOGGER = logging.getLogger(__name__)
MAX_EMPTY_RESPONSE_RECOVERY_ATTEMPTS = 2
MAX_LOW_QUALITY_RESPONSE_RECOVERY_ATTEMPTS = 2
MAX_DEFAULT_FINAL_ANSWER_CHARACTERS = 4000
MAX_DEFAULT_FINAL_ANSWER_WORDS = 150
MIN_REPEATED_PARAGRAPH_CHARACTERS = 80
MIN_CONTEXT_TOKEN_BUDGET = 8_000
AGGRESSIVE_CONTEXT_FRACTION = 0.65


@dataclass(frozen=True)
class QuestionIntent:
    primary: Literal["search", "metadata", "occurrence", "table", "visual", "location"]
    multi_document: bool = False
    multi_part: bool = False
    exhaustive: bool = False


class IngestionState(TypedDict, total=False):
    file_paths: list[str | Path]
    documents: list[Document]
    chunks: list[Chunk]
    documents_count: int
    chunks_count: int


@dataclass(frozen=True)
class IngestionContext:
    retriever: Retriever
    max_chars: int = 1000
    overlap: int = 100


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    results: list[SearchResult]
    answer: str
    search_queries: list[str]
    stop_reason: str
    documents_count: int
    chunks_count: int
    response: RAGResponse


def merge_search_results(
    current: list[SearchResult],
    new: list[SearchResult],
) -> list[SearchResult]:
    merged = list(current)
    positions = {result.chunk.id: index for index, result in enumerate(current)}
    for result in new:
        position = positions.get(result.chunk.id)
        if position is None:
            positions[result.chunk.id] = len(merged)
            merged.append(result)
        else:
            # A later read_source_context call upgrades the search preview to
            # the complete passage that was actually shown to the model. Never
            # downgrade it if the same chunk is returned by another search.
            current_result = merged[position]
            current_context = current_result.chunk.metadata.get(LLM_CONTEXT_METADATA_KEY)
            new_context = result.chunk.metadata.get(LLM_CONTEXT_METADATA_KEY)
            if not (
                isinstance(current_context, str)
                and isinstance(new_context, str)
                and len(current_context) > len(new_context)
            ):
                merged[position] = result
    return merged


def append_strings(current: list[str], new: list[str]) -> list[str]:
    return [*current, *new]


class ToolAgentState(MessagesState):
    question: str
    conversation_history: list[BaseMessage]
    sources: Annotated[list[SearchResult], merge_search_results]
    search_queries: Annotated[list[str], append_strings]
    documents_count: int
    chunks_count: int
    response: RAGResponse


@dataclass(frozen=True)
class QAContext:
    llm_client: LLMClient
    retriever: Retriever | None = None


def load_documents_node(state: IngestionState) -> dict:
    documents = load_documents(state["file_paths"])
    return {
        "documents": documents,
        "documents_count": len(documents),
    }


def chunk_documents_node(
    state: IngestionState,
    runtime: Runtime[IngestionContext],
) -> dict:
    documents = state["documents"]
    chunks = chunk_documents(
        documents=documents,
        max_chars=runtime.context.max_chars,
        overlap=runtime.context.overlap,
    )
    return {
        "chunks": chunks,
        "documents_count": len(documents),
        "chunks_count": len(chunks),
    }


def index_documents_node(
    state: IngestionState,
    runtime: Runtime[IngestionContext],
) -> dict:
    with tracer.start_as_current_span("file_agent.index_documents") as span:
        span.set_attribute("file_agent.document_count", len(state["documents"]))
        span.set_attribute("file_agent.chunk_count", len(state["chunks"]))
        runtime.context.retriever.index(state["chunks"])
    return {}


def retrieve_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    retriever = runtime.context.retriever
    if retriever is None:
        raise ValueError("A retriever is required when search results are not precomputed")

    search_query = state["question"]
    results = retriever.search(
        query=search_query,
        top_k=state.get("top_k", 5),
    )
    search_queries = list(state.get("search_queries", []))
    search_queries.append(search_query)
    return {
        "results": results,
        "search_queries": search_queries,
    }


def generate_answer_node(
    state: QAState,
    runtime: Runtime[QAContext],
) -> dict:
    answer = answer_question_with_context(
        question=state["question"],
        results=state["results"],
        llm_client=runtime.context.llm_client,
    )
    return {
        "answer": answer,
        "stop_reason": "answer_generated" if state["results"] else "no_context",
    }


def build_response_node(state: QAState) -> dict:
    return {
        "response": RAGResponse(
            answer=state["answer"],
            sources=state["results"],
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
            search_queries=list(state.get("search_queries", [])),
            retry_count=0,
            stop_reason=state.get("stop_reason", "answer_generated"),
        )
    }


def route_ingestion_input(
    state: IngestionState,
) -> Literal["load_documents", "chunk_documents"]:
    if "documents" in state:
        return "chunk_documents"
    return "load_documents"


def route_qa_input(state: QAState) -> Literal["retrieve", "generate_answer"]:
    if "results" in state:
        return "generate_answer"
    return "retrieve"


def build_ingestion_graph():
    builder = StateGraph(
        IngestionState,
        context_schema=IngestionContext,
    )
    builder.add_node("load_documents", load_documents_node)
    builder.add_node("chunk_documents", chunk_documents_node)
    builder.add_node("index_documents", index_documents_node)
    builder.add_conditional_edges(
        START,
        route_ingestion_input,
        {
            "load_documents": "load_documents",
            "chunk_documents": "chunk_documents",
        },
    )
    builder.add_edge("load_documents", "chunk_documents")
    builder.add_edge("chunk_documents", "index_documents")
    builder.add_edge("index_documents", END)
    return builder.compile(name="rag_ingestion")


def build_qa_graph():
    builder = StateGraph(
        QAState,
        context_schema=QAContext,
    )
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("build_response", build_response_node)
    builder.add_conditional_edges(
        START,
        route_qa_input,
        {
            "retrieve": "retrieve",
            "generate_answer": "generate_answer",
        },
    )
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", "build_response")
    builder.add_edge("build_response", END)
    return builder.compile(name="rag_qa")


def prepare_tool_agent_turn_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    """Build this turn's working messages from compact persisted conversation history."""
    history = _trim_conversation_history(
        list(state.get("conversation_history", [])),
        runtime.context.max_history_turns,
    )
    routing_instruction = _build_routing_instruction(state["question"], runtime.context)
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            SystemMessage(content=f"{TOOL_AGENT_SYSTEM_PROMPT}\n\n{routing_instruction}"),
            *history,
            HumanMessage(content=state["question"]),
        ],
        "conversation_history": history,
        "sources": Overwrite([]),
        "search_queries": Overwrite([]),
    }


def tool_agent_model_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    messages = list(state["messages"])
    completed_tool_rounds = _count_tool_rounds(messages)
    available_tools: Sequence[BaseTool] = (
        DOCUMENT_TOOLS if completed_tool_rounds < runtime.context.max_tool_rounds else []
    )
    if available_tools and completed_tool_rounds == 0 and runtime.context.require_evidence_tool:
        available_tools = _initial_tools_for_question(state["question"], runtime.context)
    if not available_tools:
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(
                content=f"{messages[0].content.rstrip()}\n\n{TOOL_LIMIT_MESSAGE.strip()}"
            )
        else:
            messages.insert(0, SystemMessage(content=TOOL_LIMIT_MESSAGE))

    has_document_observation = bool(state.get("sources") or state.get("search_queries"))
    tool_choice: ToolChoice = (
        "required"
        if runtime.context.require_evidence_tool
        and not has_document_observation
        and available_tools
        else "auto"
    )
    model_messages = _compact_messages_for_model(
        messages,
        max_tokens=runtime.context.max_context_tokens,
    )
    try:
        response = runtime.context.llm_client.chat_with_tools(
            messages=model_messages,
            tools=available_tools,
            tool_choice=tool_choice,
        )
    except EmptyLLMResponseError:
        response = _recover_empty_tool_agent_response(
            state=state,
            runtime=runtime,
            messages=model_messages,
            available_tools=available_tools,
        )
    except Exception as exc:
        if not _is_context_length_error(exc):
            raise
        LOGGER.warning("Tool-agent context limit reached; retrying with aggressive compaction")
        response = runtime.context.llm_client.chat_with_tools(
            messages=_compact_messages_for_model(
                messages,
                max_tokens=max(
                    MIN_CONTEXT_TOKEN_BUDGET,
                    int(runtime.context.max_context_tokens * AGGRESSIVE_CONTEXT_FRACTION),
                ),
                aggressive=True,
            ),
            tools=available_tools,
            tool_choice=tool_choice,
        )
    if not response.tool_calls and _is_low_quality_final_answer(response.content):
        response = _recover_low_quality_final_answer(
            runtime=runtime,
            messages=messages,
            initial_response=response,
        )
    if not response.tool_calls and available_tools:
        missing_files = _missing_required_evidence_files(state, runtime.context)
        if missing_files:
            response = _recover_missing_document_evidence(
                runtime=runtime,
                messages=messages,
                initial_response=response,
                missing_files=missing_files,
            )
    if not available_tools and response.tool_calls:
        raise ValueError("LLM requested a tool after the tool-call limit was reached")
    return {"messages": [response]}


def _missing_required_evidence_files(
    state: ToolAgentState,
    context: ToolAgentContext,
) -> list[str]:
    observed_files = {
        source_file.casefold()
        for result in state.get("sources", [])
        if isinstance((source_file := result.chunk.metadata.get("source_file")), str)
    }
    return [
        source_file
        for source_file in context.required_evidence_files
        if source_file.casefold() not in observed_files
    ]


def _recover_missing_document_evidence(
    *,
    runtime: Runtime[ToolAgentContext],
    messages: list[BaseMessage],
    initial_response: AIMessage,
    missing_files: list[str],
) -> AIMessage:
    LOGGER.warning(
        "Tool-agent draft omitted evidence from required document(s): %s",
        ", ".join(missing_files),
    )
    response = runtime.context.llm_client.chat_with_tools(
        messages=[
            *messages,
            initial_response,
            HumanMessage(
                content=(
                    "Your draft was not accepted because this multi-document question "
                    "still has no evidence from: "
                    f"{', '.join(missing_files)}. Call a document evidence tool now "
                    "with source_file set to one of those exact names. Do not return "
                    "the final answer until every named file has been checked."
                )
            ),
        ],
        tools=DOCUMENT_TOOLS,
        tool_choice="required",
    )
    if not response.tool_calls:
        LOGGER.warning(
            "LLM ignored the required tool choice while collecting multi-document evidence"
        )
    return response


def _is_low_quality_final_answer(content: str | list) -> bool:
    answer = str(content).strip()
    if len(answer) > MAX_DEFAULT_FINAL_ANSWER_CHARACTERS:
        return True
    if len(re.findall(r"\b[\w-]+\b", answer, flags=re.UNICODE)) > MAX_DEFAULT_FINAL_ANSWER_WORDS:
        return True

    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip().casefold()
        for paragraph in re.split(r"\n\s*\n", answer)
    ]
    substantial = [
        paragraph for paragraph in paragraphs if len(paragraph) >= MIN_REPEATED_PARAGRAPH_CHARACTERS
    ]
    return len(substantial) != len(set(substantial))


def _recover_low_quality_final_answer(
    *,
    runtime: Runtime[ToolAgentContext],
    messages: list[BaseMessage],
    initial_response: AIMessage,
) -> AIMessage:
    candidates = [initial_response]
    for attempt in range(1, MAX_LOW_QUALITY_RESPONSE_RECOVERY_ATTEMPTS + 1):
        LOGGER.warning(
            "Tool-agent LLM returned an excessively long or repetitive final answer; "
            "recovery attempt %s/%s",
            attempt,
            MAX_LOW_QUALITY_RESPONSE_RECOVERY_ATTEMPTS,
        )
        retry_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Your draft final answer was rejected because it was excessively "
                    "long or repetitive. Return only a corrected final answer of at "
                    "most 80 words using the document evidence already provided. "
                    "State one resolved conclusion, omit analysis and discarded "
                    "alternatives, and do not repeat any point. Do not call tools."
                )
            ),
        ]
        try:
            response = runtime.context.llm_client.chat_with_tools(
                messages=retry_messages,
                tools=[],
                tool_choice="auto",
            )
        except EmptyLLMResponseError:
            continue
        if response.tool_calls:
            LOGGER.warning("LLM requested a tool while rewriting its final answer")
            continue
        candidates.append(response)
        if not _is_low_quality_final_answer(response.content):
            return response

    # A bad draft should not abort a long batch. The shortest non-empty candidate
    # is the least harmful fallback and normally comes from one of the rewrites.
    return min(candidates, key=lambda candidate: len(str(candidate.content).strip()))


def _recover_empty_tool_agent_response(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
    messages: list[BaseMessage],
    available_tools: Sequence[BaseTool],
) -> AIMessage:
    has_document_observation = bool(state.get("sources") or state.get("search_queries"))
    can_call_tool = bool(available_tools) and not has_document_observation
    retry_tools = available_tools if can_call_tool else []
    retry_tool_choice: ToolChoice = "required" if can_call_tool else "auto"
    action = (
        "Return the final answer now using the document evidence already provided. "
        "If the evidence is insufficient, explicitly say so. Do not call another tool."
        if has_document_observation
        else "Call an available document tool before answering."
    )
    if not available_tools and not has_document_observation:
        action = (
            "The tool-call limit has been reached. Return the best final answer now; "
            "if no document evidence is available, explicitly say that the documents "
            "do not contain enough information. Do not call a tool."
        )

    for attempt in range(1, MAX_EMPTY_RESPONSE_RECOVERY_ATTEMPTS + 1):
        LOGGER.warning(
            "Tool-agent LLM returned an empty response; recovery attempt %s/%s",
            attempt,
            MAX_EMPTY_RESPONSE_RECOVERY_ATTEMPTS,
        )
        retry_messages = [
            *messages,
            HumanMessage(
                content=(
                    "Your previous response was empty and could not be used. "
                    f"{action} Return a non-empty response. Recovery attempt {attempt}."
                )
            ),
        ]
        try:
            response = runtime.context.llm_client.chat_with_tools(
                messages=retry_messages,
                tools=retry_tools,
                tool_choice=retry_tool_choice,
            )
            if not retry_tools and response.tool_calls:
                raise ValueError("LLM requested a tool while recovering a final response")
            return response
        except EmptyLLMResponseError:
            if attempt == MAX_EMPTY_RESPONSE_RECOVERY_ATTEMPTS:
                raise

    raise AssertionError("unreachable")


def _classify_question(question: str, context: ToolAgentContext | None = None) -> QuestionIntent:
    normalized = " ".join(question.casefold().split())
    table_terms = (
        "таблиц",
        "столб",
        "строк",
        "итог",
        "максим",
        "миним",
        "средн",
    )
    visual_terms = (
        "chart",
        "diagram",
        "figure",
        "image",
        "photo",
        "screenshot",
        "график",
        "диаграм",
        "рисунк",
        "изображен",
        "фото",
        "визуал",
        "signature",
        "stamp",
        "seal",
        "подпис",
        "печать",
    )
    occurrence_terms = (
        "how many times",
        "number of mentions",
        "occurrences",
        "сколько раз",
        "упоминан",
        "на каких страницах встречается",
    )
    metadata_terms = (
        "page count",
        "number of pages",
        "how many pages",
        "word count",
        "file size",
        "author",
        "created",
        "modified",
        "сколько страниц",
        "количество страниц",
        "количество слов",
        "размер файла",
        "автор",
        "дата создания",
        "дата изменения",
        "сколько раздел",
        "сколько рисун",
        "сколько изображ",
        "how many figures",
        "how many images",
        "сколько ссыл",
        "сколько сносок",
        "аббревиатур",
        "first sentence",
        "первое предложение",
    )
    location_terms = (
        "on page",
        "page ",
        "slide ",
        "sheet ",
        "на странице",
        "странице ",
        "слайд ",
        "лист ",
    )
    if any(term in normalized for term in occurrence_terms):
        primary = "occurrence"
    elif any(term in normalized for term in metadata_terms):
        primary = "metadata"
    elif any(term in normalized for term in table_terms) or re.search(
        r"\b(?:table|spreadsheet|sheet|column|row|rows|maximum|minimum|largest|smallest|average|sum|total)\b",
        normalized,
    ):
        primary = "table"
    elif any(term in normalized for term in visual_terms):
        primary = "visual"
    elif any(term in normalized for term in location_terms) and re.search(r"\d", normalized):
        primary = "location"
    else:
        primary = "search"

    required_files = context.required_evidence_files if context is not None else ()
    if primary == "search" and context is not None:
        target_names = {name.casefold() for name in required_files}
        target_documents = [
            document
            for document in context.documents
            if not target_names or document.file_name.casefold() in target_names
        ]
        if target_documents and all(
            document.file_type.casefold().lstrip(".") == "xlsx" for document in target_documents
        ):
            primary = "table"
    multi_document = len(required_files) > 1
    multi_part = bool(
        re.search(r"(?:\?|;).+", normalized)
        or re.search(r"\b(?:and|also|both|и|а также|оба|обе)\b", normalized)
    )
    exhaustive = any(
        term in normalized
        for term in ("all ", "every ", "complete list", "все ", "кажд", "полный список", "сколько")
    )
    return QuestionIntent(primary, multi_document, multi_part, exhaustive)


def _initial_tools_for_question(question: str, context: ToolAgentContext) -> list[BaseTool]:
    intent = _classify_question(question, context)
    names_by_intent = {
        "search": {"search_documents"},
        "metadata": {"get_document_metadata"},
        "occurrence": {"count_document_occurrences"},
        "table": {"get_document_outline"},
        "visual": {"get_document_outline"},
        "location": {"read_document_location"},
    }
    selected_names = names_by_intent[intent.primary]
    return [tool for tool in DOCUMENT_TOOLS if tool.name in selected_names]


def _build_routing_instruction(question: str, context: ToolAgentContext) -> str:
    intent = _classify_question(question, context)
    file_names = [document.file_name for document in context.documents]
    lines = [
        "Turn-specific evidence plan (follow this before drafting):",
        f"- Available exact source_file values: {file_names}.",
    ]
    route = {
        "search": "Start with source-filtered search_documents and expand incomplete passages.",
        "metadata": "Start with get_document_metadata; use its deterministic content statistics.",
        "occurrence": "Start with count_document_occurrences over the complete requested scope.",
        "table": (
            "Start with get_document_outline, then use lookup_table_rows or analyze_table. "
            "Do not use a semantic-search preview as the final table evidence."
        ),
        "visual": (
            "Start with get_document_outline, select the exact visual, then call "
            "analyze_document_visual before making visual claims."
        ),
        "location": "Start with read_document_location for the explicitly named page/slide/sheet.",
    }[intent.primary]
    lines.append(f"- {route}")
    if intent.multi_document:
        lines.append(
            "- This is multi-document: collect a separate, source-filtered evidence item from "
            f"each required file: {list(context.required_evidence_files)}. Do not merge the files "
            "into one unsupported conclusion."
        )
    if intent.multi_part or intent.exhaustive:
        lines.append(
            "- Before answering, audit every requested clause/item against an explicit "
            "tool result. If one is unsupported, say exactly which part is unavailable "
            "instead of guessing."
        )
    lines.append(
        "- Reject proxy inferences: metadata dates are not revision claims, totals are not salary, "
        "proximity is not a mapping, and failed retrieval alone is not proof of absence."
    )
    return "\n".join(lines)


def _estimate_text_tokens(text: str) -> int:
    ascii_chars = sum(1 for character in text if ord(character) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + (non_ascii_chars + 1) // 2)


def _estimate_messages_tokens(messages: Sequence[BaseMessage]) -> int:
    total = 0
    for message in messages:
        total += 8 + _estimate_text_tokens(str(message.content))
        if isinstance(message, AIMessage) and message.tool_calls:
            total += _estimate_text_tokens(str(message.tool_calls))
    return total


def _compact_messages_for_model(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    aggressive: bool = False,
) -> list[BaseMessage]:
    if max_tokens < MIN_CONTEXT_TOKEN_BUDGET:
        raise ValueError(f"max context token budget must be at least {MIN_CONTEXT_TOKEN_BUDGET}")
    compacted = list(messages)
    if _estimate_messages_tokens(compacted) <= max_tokens:
        return compacted

    seen_tool_payloads: set[str] = set()
    per_message_chars = 2_000 if aggressive else 6_000
    for index, message in enumerate(compacted):
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content)
        if content in seen_tool_payloads:
            replacement = "[Duplicate tool evidence omitted; use the earlier identical result.]"
        else:
            seen_tool_payloads.add(content)
            replacement = _truncate_evidence_content(content, per_message_chars)
        compacted[index] = message.model_copy(update={"content": replacement})

    if _estimate_messages_tokens(compacted) <= max_tokens:
        return compacted

    # Preserve tool-call/result protocol pairs, but shrink every observation to a
    # compact evidence card. System instructions and the current question remain intact.
    for index, message in enumerate(compacted):
        if isinstance(message, ToolMessage):
            compacted[index] = message.model_copy(
                update={"content": _truncate_evidence_content(str(message.content), 1_200)}
            )
    return compacted


def _truncate_evidence_content(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    tail_chars = min(400, max_chars // 4)
    head_chars = max_chars - tail_chars
    return (
        content[:head_chars].rstrip()
        + "\n[...older/verbose evidence compacted...]\n"
        + content[-tail_chars:].lstrip()
    )


def _is_context_length_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(
        marker in message
        for marker in (
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "input tokens",
            "context window",
        )
    )


def route_tool_agent(
    state: ToolAgentState,
) -> Literal["tools", "build_response"]:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "build_response"


def build_tool_agent_response_node(
    state: ToolAgentState,
    runtime: Runtime[ToolAgentContext],
) -> dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or last_message.tool_calls:
        raise ValueError("Tool agent did not return a final answer")

    answer = str(last_message.content).strip()
    tool_calls = _collect_tool_calls(state["messages"])
    if (
        runtime.context.require_evidence_tool
        and not state.get("sources")
        and not state.get("search_queries")
    ):
        raise ValueError(
            "Tool agent returned a final answer without successfully using a document evidence tool"
        )
    search_queries = list(state.get("search_queries", []))
    history = _trim_conversation_history(
        [
            *state.get("conversation_history", []),
            HumanMessage(content=state["question"]),
            AIMessage(content=answer),
        ],
        runtime.context.max_history_turns,
    )
    return {
        "response": RAGResponse(
            answer=answer,
            sources=list(state.get("sources", [])),
            documents_count=state.get("documents_count", 0),
            chunks_count=state.get("chunks_count", 0),
            search_queries=search_queries,
            retry_count=max(0, len(search_queries) - 1),
            stop_reason="tool_agent_completed",
            tool_calls=tool_calls,
        ),
        "conversation_history": history,
        # Tool calls and observations remain available for the full current run,
        # then are removed from the latest checkpoint. Only compact Q/A pairs live
        # into the next turn.
        "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
    }


def build_tool_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(
        ToolAgentState,
        context_schema=ToolAgentContext,
    )
    builder.add_node("prepare_turn", prepare_tool_agent_turn_node)
    builder.add_node("agent_model", tool_agent_model_node)
    builder.add_node(
        "tools",
        ToolNode(DOCUMENT_TOOLS, handle_tool_errors=True),
    )
    builder.add_node("build_response", build_tool_agent_response_node)
    builder.add_edge(START, "prepare_turn")
    builder.add_edge("prepare_turn", "agent_model")
    builder.add_conditional_edges(
        "agent_model",
        route_tool_agent,
        {
            "tools": "tools",
            "build_response": "build_response",
        },
    )
    builder.add_edge("tools", "agent_model")
    builder.add_edge("build_response", END)
    return builder.compile(name="rag_tool_agent", checkpointer=checkpointer)


def _trim_conversation_history(
    messages: list[BaseMessage],
    max_history_turns: int,
) -> list[BaseMessage]:
    if max_history_turns < 1:
        raise ValueError("max_history_turns must be greater than zero")

    # Persist only complete user/final-assistant pairs. Tool calls, ToolMessages,
    # system instructions, and any malformed fragments are intentionally omitted.
    pairs: list[tuple[HumanMessage, AIMessage]] = []
    pending_user: HumanMessage | None = None
    for message in messages:
        if isinstance(message, HumanMessage):
            pending_user = message
        elif isinstance(message, AIMessage) and not message.tool_calls and pending_user is not None:
            pairs.append((pending_user, message))
            pending_user = None

    trimmed_pairs = pairs[-max_history_turns:]
    return [message for pair in trimmed_pairs for message in pair]


def _count_tool_rounds(messages: list[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, AIMessage) and message.tool_calls)


def _collect_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    collected = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls:
            collected.append(
                {
                    "name": tool_call["name"],
                    "arguments": dict(tool_call["args"]),
                }
            )
    return collected


ingestion_graph = build_ingestion_graph()
qa_graph = build_qa_graph()
tool_agent_checkpointer = InMemorySaver()
stateless_tool_agent_graph = build_tool_agent_graph()
tool_agent_graph = build_tool_agent_graph(checkpointer=tool_agent_checkpointer)
