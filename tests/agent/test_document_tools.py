"""Test document tools."""

import json

import pytest
from langchain_core.messages import AIMessage

from file_agent.agent_tools import MAX_TOOL_CONTENT_LENGTH
from file_agent.document import Block, BlockType, Document
from tests.agent._support import (
    FakeRetriever,
    FakeToolCallingLLM,
    _invoke_single_read_tool,
    invoke_tool_agent,
    make_metadata_document,
    make_occurrence_document,
    make_structured_document,
    tool_messages_seen_by_model,
)


def test_required_metadata_question_exposes_only_metadata_tool_first():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_metadata",
                        "args": {"source_file": "annual-report.pdf"},
                        "id": "metadata-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The document has four pages."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[make_metadata_document()],
        require_evidence_tool=True,
        question="How many pages are in the document?",
    )

    assert state["response"].answer == "The document has four pages."
    assert llm_client.calls[0]["tool_names"] == ["get_document_metadata"]
    assert "Turn-specific evidence plan" in llm_client.calls[0]["messages"][0].content


def test_tool_agent_can_navigate_outline_and_read_section():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "handbook.pdf"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_document_section",
                        "args": {
                            "source_file": "handbook.pdf",
                            "section_id": "heading-main",
                        },
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The main section includes its child section."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[make_structured_document()],
    )

    tool_messages = tool_messages_seen_by_model(llm_client)
    outline = json.loads(str(tool_messages[0].content))
    section = json.loads(str(tool_messages[1].content))
    assert outline["table_of_contents"][0]["section_id"] == "heading-main"
    assert outline["tables"][0]["table_id"] == "table-1"
    assert outline["locations"] == {"slides": [], "sheets": []}
    assert "Main section body." in section["text"]
    assert "Child section body." in section["text"]
    assert "must not be returned" not in section["text"]
    assert state["response"].sources[0].chunk.metadata["page_numbers"] == [1, 2]


def test_document_outline_includes_presentation_slide_map():
    document = Document(
        file_name="deck.pptx",
        file_type="pptx",
        blocks=[
            Block(
                id="slide-1",
                text="Slide 1\nText:\nPresentation title",
                type="pptx_slide",
                metadata={"slide_number": 1},
            ),
            Block(
                id="slide-2",
                text="Slide 2\nTitle: Main topic\nText:\nDetails",
                type="pptx_slide",
                metadata={"slide_number": 2},
            ),
        ],
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "deck.pptx"},
                        "id": "outline-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deck has two slides."),
        ]
    )

    invoke_tool_agent(llm_client, FakeRetriever(), documents=[document])

    outline = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert outline["locations"]["slides"] == [
        {"slide_number": 1, "title": "Presentation title"},
        {"slide_number": 2, "title": "Main topic"},
    ]


def test_document_outline_identifies_numbered_presentation_sections():
    document = Document(
        file_name="deck.pptx",
        file_type="pptx",
        blocks=[
            Block(
                id="slide-1",
                text="Slide 1\nText:\nPresentation title",
                type="pptx_slide",
                metadata={"slide_number": 1},
            ),
            Block(
                id="slide-2",
                text="Slide 2\nText:\nCommunication models\n1",
                type="pptx_slide",
                metadata={"slide_number": 2},
            ),
            Block(
                id="slide-3",
                text="Slide 3\nText:\nModel details\nNot a divider",
                type="pptx_slide",
                metadata={"slide_number": 3},
            ),
        ],
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "deck.pptx"},
                        "id": "outline-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deck has one numbered section."),
        ]
    )

    invoke_tool_agent(llm_client, FakeRetriever(), documents=[document])

    outline = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert outline["locations"]["numbered_sections"] == [
        {"section_number": 1, "title": "Communication models", "slide_number": 2}
    ]
    assert outline["locations"]["thematic_section_review"] == {
        "opening_numbered_candidates": [
            {"section_number": 1, "title": "Communication models", "slide_number": 2}
        ],
        "substantial_unnumbered_candidates": [],
        "guidance": (
            "Do not copy the numbered-divider count mechanically. Inspect opening "
            "candidates and exclude them when they are introductory; include each "
            "substantial unnumbered topic, then list the final thematic sections."
        ),
    }


def test_document_outline_flags_substantial_unnumbered_presentation_topic():
    document = Document(
        file_name="deck.pptx",
        file_type="pptx",
        blocks=[
            Block(
                id="slide-1",
                text="Slide 1\nText:\nPresentation title",
                type="pptx_slide",
                metadata={"slide_number": 1},
            ),
            Block(
                id="slide-2",
                text="Slide 2\nText:\nIntroductory overview\n1",
                type="pptx_slide",
                metadata={"slide_number": 2},
            ),
            Block(
                id="slide-3",
                text="Slide 3\nText:\nCommunication barriers\nExamples",
                type="pptx_slide",
                metadata={"slide_number": 3},
            ),
        ],
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "deck.pptx"},
                        "id": "outline-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The deck has one thematic section."),
        ]
    )

    invoke_tool_agent(llm_client, FakeRetriever(), documents=[document])

    outline = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert outline["locations"]["thematic_section_review"]["substantial_unnumbered_candidates"] == [
        {"slide_number": 3, "title": "Communication barriers"}
    ]
    assert outline["locations"]["thematic_section_review"]["introductory_numbered_sections"] == [
        {"section_number": 1, "title": "Introductory overview", "slide_number": 2}
    ]
    assert outline["locations"]["thematic_section_review"]["recommended_thematic_sections"] == [
        {"slide_number": 3, "title": "Communication barriers"}
    ]


def test_direct_read_source_preserves_dataset_document_id():
    document = make_structured_document()
    for block in document.blocks:
        block.metadata["dataset_doc_id"] = "q0001/handbook.pdf"
        block.metadata["dataset_record_id"] = "q0001"

    state, _ = _invoke_single_read_tool(
        "read_document_location",
        {"source_file": "handbook.pdf", "page_number": 3},
        [document],
    )

    metadata = state["response"].sources[0].chunk.metadata
    assert metadata["dataset_doc_id"] == "q0001/handbook.pdf"
    assert metadata["dataset_record_id"] == "q0001"


def test_document_metadata_reports_exact_full_document_statistics():
    document = make_metadata_document()

    state, llm_client = _invoke_single_read_tool(
        "get_document_metadata",
        {"source_file": "annual-report.pdf"},
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_pages"] == 4
    assert payload["file_size_bytes"] == 12345
    assert payload["pdf_metadata"] == {
        "title": "2026 Annual Report",
        "author": "Ada Lovelace",
    }
    assert payload["structure"] == {
        "scope": "full_document",
        "page_number": None,
        "blocks": 3,
        "non_empty_blocks": 3,
        "headings": 1,
        "top_level_sections": [{"title": "Annual Report", "page_number": 1}],
        "tables": 0,
        "visuals": 1,
        "content_visuals": 1,
        "numbered_figures": 0,
        "pages_with_extracted_blocks": [1, 2],
    }
    assert payload["content_statistics"]["first_content_sentence"] == (
        "Alpha beta state-of-the-art."
    )
    assert payload["text_statistics"]["word_count"] == 8
    assert payload["text_statistics"]["extracted_word_count"] == 8
    assert payload["text_statistics"]["native_pdf_text_layer_word_count"] == 6
    assert payload["text_statistics"]["native_zero_text_pages_using_parsed_fallback"] == [2]
    assert "generated by the VLM" not in state["response"].sources[0].chunk.text
    assert state["response"].sources[0].chunk.metadata == {
        "source_file": "annual-report.pdf",
        "file_type": "pdf",
        "dataset_doc_id": "docbench/0.pdf",
        "evidence_type": "document_metadata",
    }


def test_document_metadata_can_count_words_on_one_page():
    state, llm_client = _invoke_single_read_tool(
        "get_document_metadata",
        {"source_file": "annual-report.pdf", "page_number": 2},
        [make_metadata_document()],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["text_statistics"]["scope"] == "page"
    assert payload["text_statistics"]["page_number"] == 2
    assert payload["text_statistics"]["word_count"] == 2
    assert payload["text_statistics"]["word_count_source"] == "parsed_document_fallback"
    assert state["response"].sources[0].chunk.metadata["page_number"] == 2


def test_occurrence_count_scans_every_block_and_returns_pages():
    document = make_occurrence_document()

    state, llm_client = _invoke_single_read_tool(
        "count_document_occurrences",
        {
            "source_file": "memory.pdf",
            "query": "memory",
            "match_mode": "whole_word",
            "max_examples": 2,
        },
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_occurrences"] == 5
    assert payload["matched_blocks"] == 2
    assert payload["matched_pages"] == [1, 2]
    assert len(payload["examples"]) == 2
    assert payload["examples_truncated"] is True
    source = state["response"].sources[0]
    assert source.chunk.metadata["total_occurrences"] == 5
    assert source.chunk.metadata["page_numbers"] == [1, 2]
    assert source.chunk.metadata["dataset_doc_id"] == "docbench/memory.pdf"


def test_occurrence_count_matches_phrases_across_whitespace():
    _, llm_client = _invoke_single_read_tool(
        "count_document_occurrences",
        {
            "source_file": "memory.pdf",
            "query": "agentic memory",
            "match_mode": "whole_word",
        },
        [make_occurrence_document()],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_occurrences"] == 2
    assert payload["matched_pages"] == [1, 2]


def test_occurrence_count_supports_page_filter_and_case_sensitivity():
    _, page_llm_client = _invoke_single_read_tool(
        "count_document_occurrences",
        {
            "source_file": "memory.pdf",
            "query": "memory",
            "page_number": 2,
        },
        [make_occurrence_document()],
    )
    page_payload = json.loads(str(tool_messages_seen_by_model(page_llm_client)[0].content))
    assert page_payload["total_occurrences"] == 2
    assert page_payload["matched_pages"] == [2]

    _, case_llm_client = _invoke_single_read_tool(
        "count_document_occurrences",
        {
            "source_file": "memory.pdf",
            "query": "memory",
            "case_sensitive": True,
        },
        [make_occurrence_document()],
    )
    case_payload = json.loads(str(tool_messages_seen_by_model(case_llm_client)[0].content))
    assert case_payload["total_occurrences"] == 4


def test_occurrence_count_zero_is_still_document_evidence():
    state, llm_client = _invoke_single_read_tool(
        "count_document_occurrences",
        {"source_file": "memory.pdf", "query": "absent phrase"},
        [make_occurrence_document()],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_occurrences"] == 0
    assert payload["matched_pages"] == []
    assert state["response"].sources[0].chunk.metadata["dataset_doc_id"] == ("docbench/memory.pdf")


def test_tool_agent_can_read_an_unstructured_document_with_pagination():
    text = "Start of document. " + "x" * MAX_TOOL_CONTENT_LENGTH + " End of document."
    document = Document(
        file_name="notes.txt",
        file_type="txt",
        blocks=[
            Block(
                id="block-1",
                text=text,
                type="plain_text",
                metadata={"dataset_doc_id": "q1/notes.txt"},
            )
        ],
    )

    state, llm_client = _invoke_single_read_tool(
        "read_document",
        {"source_file": "notes.txt"},
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["text"].startswith("Start of document.")
    assert len(payload["text"]) <= MAX_TOOL_CONTENT_LENGTH
    assert payload["next_offset"] is not None
    assert payload["total_characters"] == len(text)
    source = state["response"].sources[0]
    assert source.chunk.id == "document:notes.txt:0"
    assert source.chunk.metadata["evidence_type"] == "document_read"
    assert source.chunk.metadata["dataset_doc_id"] == "q1/notes.txt"

    _, next_llm_client = _invoke_single_read_tool(
        "read_document",
        {
            "source_file": "notes.txt",
            "offset": payload["next_offset"],
        },
        [document],
    )
    next_payload = json.loads(str(tool_messages_seen_by_model(next_llm_client)[0].content))
    assert next_payload["text"].endswith("End of document.")
    assert next_payload["next_offset"] is None


@pytest.mark.parametrize(
    ("documents", "arguments", "expected_text", "metadata_key", "metadata_value"),
    [
        (
            [make_structured_document()],
            {"source_file": "handbook.pdf", "page_number": 3},
            "Next section",
            "page_number",
            3,
        ),
        (
            [
                Document(
                    file_name="deck.pptx",
                    file_type="pptx",
                    blocks=[
                        Block(
                            id="slide-2",
                            text="Slide 2 metrics",
                            type="pptx_slide",
                            metadata={"slide_number": 2},
                        )
                    ],
                )
            ],
            {"source_file": "deck.pptx", "slide_number": 2},
            "Slide 2 metrics",
            "slide_number",
            2,
        ),
        (
            [
                Document(
                    file_name="budget.xlsx",
                    file_type="xlsx",
                    blocks=[
                        Block(
                            id="sheet-1",
                            text="Month\tAmount\nJanuary\t100",
                            type="xlsx_sheet",
                            metadata={"sheet_name": "Budget"},
                            block_type=BlockType.TABLE,
                        )
                    ],
                )
            ],
            {"source_file": "budget.xlsx", "sheet_name": "budget"},
            "January",
            "sheet_name",
            "Budget",
        ),
    ],
)
def test_tool_agent_can_read_document_locations(
    documents,
    arguments,
    expected_text,
    metadata_key,
    metadata_value,
):
    state, llm_client = _invoke_single_read_tool("read_document_location", arguments, documents)

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert expected_text in payload["text"]
    assert payload["location"][metadata_key] == metadata_value
    assert state["response"].sources[0].chunk.metadata[metadata_key] == metadata_value
