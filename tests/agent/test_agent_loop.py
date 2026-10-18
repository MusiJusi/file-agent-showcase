"""Test agent loop."""

import json

import pytest
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from file_agent.agent_tools import (
    DOCUMENT_TOOLS,
    calculate,
    count_document_occurrences,
    get_document_metadata,
    search_documents,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, Document
from file_agent.llm.base import EmptyLLMResponseError
from file_agent.rag_graph import build_tool_agent_graph
from file_agent.retrieval import SearchResult
from tests.agent._support import (
    FakeRetriever,
    FakeToolCallingLLM,
    invoke_persistent_turn,
    invoke_tool_agent,
    make_document,
    make_search_result,
    tool_messages_seen_by_model,
)


@pytest.mark.parametrize("document_tool", DOCUMENT_TOOLS)
def test_document_tool_schemas_hide_injected_runtime(document_tool):
    schema = document_tool.tool_call_schema.model_json_schema()

    assert "runtime" not in schema["properties"]


def test_search_tool_exposes_filter_schema():
    schema = search_documents.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {"query", "top_k", "source_file"}


def test_new_tool_schemas_are_small_and_explicit():
    read_tool = next(tool for tool in DOCUMENT_TOOLS if tool.name == "read_document")

    assert set(read_tool.tool_call_schema.model_json_schema()["properties"]) == {
        "source_file",
        "offset",
    }
    assert set(calculate.tool_call_schema.model_json_schema()["properties"]) == {
        "operation",
        "values",
    }
    assert set(get_document_metadata.tool_call_schema.model_json_schema()["properties"]) == {
        "source_file",
        "page_number",
    }
    assert set(count_document_occurrences.tool_call_schema.model_json_schema()["properties"]) == {
        "source_file",
        "query",
        "match_mode",
        "case_sensitive",
        "page_number",
        "max_examples",
    }


def test_tool_agent_searches_documents_and_returns_sources():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline", "top_k": 3},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The project deadline is Friday [plan.md, page 2]."),
        ]
    )
    retriever = FakeRetriever([make_search_result()])

    state = invoke_tool_agent(llm_client, retriever)

    response = state["response"]
    assert response.answer == "The project deadline is Friday [plan.md, page 2]."
    assert response.sources[0].chunk.id == make_search_result().chunk.id
    assert response.sources[0].chunk.metadata["_llm_context"] == ("The project deadline is Friday.")
    assert response.search_queries == ["project deadline"]
    assert response.stop_reason == "tool_agent_completed"
    assert response.tool_calls == [
        {
            "name": "search_documents",
            "arguments": {"query": "project deadline", "top_k": 3},
        }
    ]
    assert retriever.search_calls == [("project deadline", 3, None)]
    assert tool_messages_seen_by_model(llm_client)
    assert state["messages"] == []
    assert llm_client.responses == []


def test_tool_agent_recovers_an_empty_response_after_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            EmptyLLMResponseError("LLM returned an empty response"),
            AIMessage(content="The project deadline is Friday [plan.md, page 2]."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([make_search_result()]))

    assert state["response"].answer == "The project deadline is Friday [plan.md, page 2]."
    assert len(llm_client.calls) == 3
    assert llm_client.calls[-1]["tool_names"] == []
    assert "Return the final answer now" in llm_client.calls[-1]["messages"][-1].content


def test_tool_agent_rewrites_an_excessively_long_final_answer():
    repetitive_answer = ("The project deadline is Friday. " * 160).strip()
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=repetitive_answer),
            AIMessage(content="The project deadline is Friday [plan.md, page 2]."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([make_search_result()]))

    assert state["response"].answer == "The project deadline is Friday [plan.md, page 2]."
    assert len(llm_client.calls) == 3
    assert llm_client.calls[-1]["tool_names"] == []
    assert "at most 80 words" in llm_client.calls[-1]["messages"][-1].content


def test_tool_agent_rewrites_repeated_paragraphs_in_a_shorter_answer():
    repeated_paragraph = (
        "The evidence explicitly says the deadline is Friday and identifies the "
        "project schedule as the source."
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=f"{repeated_paragraph}\n\n{repeated_paragraph}"),
            AIMessage(content="The project deadline is Friday."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([make_search_result()]))

    assert state["response"].answer == "The project deadline is Friday."
    assert len(llm_client.calls) == 3


def test_tool_agent_can_require_a_document_evidence_tool():
    llm_client = FakeToolCallingLLM([AIMessage(content="An unsupported direct answer.")])

    with pytest.raises(ValueError, match="without successfully using"):
        invoke_tool_agent(
            llm_client,
            FakeRetriever(),
            require_evidence_tool=True,
        )

    assert llm_client.calls[0]["tool_choice"] == "required"


def test_required_document_evidence_returns_to_auto_after_a_tool_result():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The project deadline is Friday."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([make_search_result()]),
        require_evidence_tool=True,
    )

    assert state["response"].answer == "The project deadline is Friday."
    assert [call["tool_choice"] for call in llm_client.calls] == ["required", "auto"]


def test_multi_document_evaluation_collects_evidence_from_every_required_file():
    second_document = Document(
        file_name="details.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="details-block",
                text="The detailed budget is 42 units.",
                type="text",
                metadata={"source_file": "details.pdf", "page_number": 3},
            )
        ],
    )
    second_result = SearchResult(
        chunk=Chunk(
            id="details-chunk",
            text="detailed budget",
            metadata={
                "context": "The detailed budget is 42 units.",
                "source_file": "details.pdf",
                "page_number": 3,
            },
        ),
        score=0.8,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "deadline", "source_file": "plan.md"},
                        "id": "search-plan",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deadline is Friday."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "budget", "source_file": "details.pdf"},
                        "id": "search-details",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deadline is Friday and the budget is 42 units."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([make_search_result(), second_result]),
        documents=[make_document(), second_document],
        require_evidence_tool=True,
        required_evidence_files=("plan.md", "details.pdf"),
    )

    assert state["response"].answer == ("The deadline is Friday and the budget is 42 units.")
    assert [call["tool_choice"] for call in llm_client.calls] == [
        "required",
        "auto",
        "required",
        "auto",
    ]
    assert {result.chunk.metadata["source_file"] for result in state["response"].sources} == {
        "plan.md",
        "details.pdf",
    }


def test_required_evidence_allows_a_successful_search_with_no_matches():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "missing fact"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The documents do not contain enough information."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        require_evidence_tool=True,
    )

    assert state["response"].sources == []
    assert state["response"].search_queries == ["missing fact"]


def test_tool_agent_can_list_documents_without_retrieval():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_documents",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The indexed document is plan.md."),
        ]
    )
    retriever = FakeRetriever()

    state = invoke_tool_agent(llm_client, retriever)

    tool_messages = tool_messages_seen_by_model(llm_client)
    assert "plan.md" in str(tool_messages[0].content)
    assert state["response"].sources == []
    assert retriever.search_calls == []


def test_tool_agent_forces_final_answer_after_tool_round_limit():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "project deadline", "top_k": 5},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The available context says the deadline is Friday."),
        ]
    )
    retriever = FakeRetriever([make_search_result()])

    state = invoke_tool_agent(llm_client, retriever, max_tool_rounds=1)

    assert state["response"].answer.endswith("deadline is Friday.")
    assert llm_client.calls[0]["tool_names"] == [tool.name for tool in DOCUMENT_TOOLS]
    assert llm_client.calls[1]["tool_names"] == []
    final_messages = llm_client.calls[1]["messages"]
    assert isinstance(final_messages[0], SystemMessage)
    assert "uploaded documents" in str(final_messages[0].content)
    assert "tool-call limit" in str(final_messages[0].content)
    assert sum(isinstance(message, SystemMessage) for message in final_messages) == 1
    assert retriever.search_calls == [("project deadline", 5, None)]


def test_empty_response_at_tool_limit_retries_without_required_empty_toolset():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "list_documents",
                        "args": {},
                        "id": "list-1",
                        "type": "tool_call",
                    }
                ],
            ),
            EmptyLLMResponseError("LLM returned an empty response"),
            AIMessage(content="There is not enough document evidence to answer."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever(), max_tool_rounds=1)

    assert state["response"].answer == "There is not enough document evidence to answer."
    assert llm_client.calls[-1]["tool_names"] == []
    assert llm_client.calls[-1]["tool_choice"] == "auto"


def test_large_tool_observation_is_compacted_before_next_model_call():
    long_context = "start " + ("important evidence " * 8_000) + " final value 42"
    result = SearchResult(
        chunk=Chunk(
            id="large-context",
            text="important evidence",
            metadata={"context": long_context, "source_file": "plan.md"},
        ),
        score=1.0,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "important evidence"},
                        "id": "search-large",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source_context",
                        "args": {"chunk_id": "large-context"},
                        "id": "read-large",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The final value is 42."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([result]),
        max_context_tokens=8_000,
    )

    final_tool_messages = [
        message for message in llm_client.calls[-1]["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(str(final_tool_messages[-1].content)) < len(long_context)
    assert "evidence compacted" in str(final_tool_messages[-1].content)
    assert state["response"].sources[0].chunk.metadata["_llm_context"] == long_context


def test_tool_agent_returns_tool_errors_to_model():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "missing.pdf"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The requested document is not indexed."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever())

    tool_messages = tool_messages_seen_by_model(llm_client)
    assert "document is not indexed" in str(tool_messages[0].content)
    assert state["response"].answer == "The requested document is not indexed."


def test_tool_agent_can_filter_search_by_source_file():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {
                            "query": "project deadline",
                            "source_file": "PLAN.MD",
                            "top_k": 3,
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deadline is Friday."),
        ]
    )
    retriever = FakeRetriever(
        [
            make_search_result(),
            SearchResult(
                chunk=Chunk(
                    id="other-chunk",
                    text="Other deadline",
                    metadata={"source_file": "other.md"},
                ),
                score=0.8,
            ),
        ]
    )

    state = invoke_tool_agent(llm_client, retriever)

    assert retriever.search_calls == [("project deadline", 3, "plan.md")]
    assert state["response"].sources[0].chunk.id == make_search_result().chunk.id
    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert payload["source_file"] == "plan.md"
    assert payload["results_count"] == 1


def test_tool_agent_can_read_full_context_after_search():
    long_context = "Beginning " + "x" * 3700 + " complete ending"
    result = SearchResult(
        chunk=Chunk(
            id="long-chunk",
            text="short match",
            metadata={"context": long_context, "source_file": "plan.md"},
        ),
        score=0.9,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "complete ending"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source_context",
                        "args": {"chunk_id": "long-chunk"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "complete ending"},
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The context has a complete ending."),
        ]
    )

    state = invoke_tool_agent(llm_client, FakeRetriever([result]))

    tool_messages = tool_messages_seen_by_model(llm_client)
    search_payload = json.loads(str(tool_messages[0].content))
    context_payload = json.loads(str(tool_messages[1].content))
    repeated_search_payload = json.loads(str(tool_messages[2].content))
    assert search_payload["results"][0]["text"].endswith("...")
    assert context_payload["text"] == long_context
    assert repeated_search_payload["results"][0]["text"].endswith("...")
    assert state["response"].sources[0].chunk.id == result.chunk.id
    assert state["response"].sources[0].chunk.metadata["_llm_context"] == long_context


def test_persistent_agent_uses_compact_history_and_retrieves_again_each_turn():
    first_result = make_search_result()
    second_result = SearchResult(
        chunk=Chunk(
            id="quarter-2",
            text="Second-quarter revenue was 150 million rubles.",
            metadata={"source_file": "plan.md", "page_number": 3},
        ),
        score=0.95,
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "first-quarter revenue"},
                        "id": "turn-1-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="First-quarter revenue was 120 million rubles."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "second-quarter revenue"},
                        "id": "turn-2-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Second-quarter revenue was 150 million rubles."),
        ]
    )
    retriever = FakeRetriever([first_result])
    checkpointer = InMemorySaver()
    graph = build_tool_agent_graph(checkpointer=checkpointer)

    first_state = invoke_persistent_turn(
        graph,
        llm_client,
        retriever,
        "What was the first-quarter revenue?",
    )
    retriever.results = [second_result]
    second_state = invoke_persistent_turn(
        graph,
        llm_client,
        retriever,
        "And in the second?",
    )

    second_turn_input = llm_client.calls[2]["messages"]
    assert [message.content for message in second_turn_input[1:]] == [
        "What was the first-quarter revenue?",
        "First-quarter revenue was 120 million rubles.",
        "And in the second?",
    ]
    assert not any(isinstance(message, ToolMessage) for message in second_turn_input)
    assert not any(
        isinstance(message, AIMessage) and message.tool_calls for message in second_turn_input
    )
    assert retriever.search_calls == [
        ("first-quarter revenue", 5, None),
        ("second-quarter revenue", 5, None),
    ]
    assert [source.chunk.id for source in first_state["sources"]] == [first_result.chunk.id]
    assert [source.chunk.id for source in second_state["sources"]] == [second_result.chunk.id]
    assert [source.chunk.id for source in second_state["response"].sources] == [
        second_result.chunk.id
    ]
    assert second_state["messages"] == []

    persisted = graph.get_state({"configurable": {"thread_id": "conversation-1"}}).values
    assert persisted["messages"] == []
    assert [message.content for message in persisted["conversation_history"]] == [
        "What was the first-quarter revenue?",
        "First-quarter revenue was 120 million rubles.",
        "And in the second?",
        "Second-quarter revenue was 150 million rubles.",
    ]


def test_persistent_agent_limits_history_to_complete_recent_pairs():
    llm_client = FakeToolCallingLLM([AIMessage(content=f"Answer {turn}") for turn in range(1, 5)])
    graph = build_tool_agent_graph(checkpointer=InMemorySaver())

    for turn in range(1, 5):
        state = invoke_persistent_turn(
            graph,
            llm_client,
            FakeRetriever(),
            f"Question {turn}",
            thread_id="bounded-history",
            max_history_turns=2,
        )

    fourth_turn_input = llm_client.calls[3]["messages"]
    assert [message.content for message in fourth_turn_input[1:]] == [
        "Question 2",
        "Answer 2",
        "Question 3",
        "Answer 3",
        "Question 4",
    ]
    assert [message.content for message in state["conversation_history"]] == [
        "Question 3",
        "Answer 3",
        "Question 4",
        "Answer 4",
    ]
    assert state["messages"] == []


def test_different_thread_id_starts_without_previous_conversation():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(content="First thread answer"),
            AIMessage(content="New thread answer"),
        ]
    )
    graph = build_tool_agent_graph(checkpointer=InMemorySaver())

    invoke_persistent_turn(
        graph,
        llm_client,
        FakeRetriever(),
        "Question in the first chat",
        thread_id="thread-one",
    )
    state = invoke_persistent_turn(
        graph,
        llm_client,
        FakeRetriever(),
        "Question in a new chat",
        thread_id="thread-two",
    )

    assert [message.content for message in llm_client.calls[1]["messages"][1:]] == [
        "Question in a new chat"
    ]
    assert [message.content for message in state["conversation_history"]] == [
        "Question in a new chat",
        "New thread answer",
    ]


def test_grounding_prompt_requires_fresh_document_evidence_for_followups():
    first_model_messages = FakeToolCallingLLM([AIMessage(content="Answer")])

    invoke_tool_agent(first_model_messages, FakeRetriever())

    system_prompt = str(first_model_messages.calls[0]["messages"][0].content)
    assert "Conversation history" in system_prompt
    assert "It is not evidence" in system_prompt
    assert "For every new user turn" in system_prompt
