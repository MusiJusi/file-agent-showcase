"""Test visual tools."""

import json

from langchain_core.messages import AIMessage

from file_agent.agent_tools import DOCUMENT_TOOLS
from file_agent.document import Block, BlockType, Document
from file_agent.document_assets import InMemoryDocumentAssetStore
from tests.agent._support import (
    FakeRetriever,
    FakeToolCallingLLM,
    RecordingVLM,
    _invoke_single_read_tool,
    invoke_tool_agent,
    make_visual_pdf,
    tool_messages_seen_by_model,
)


def test_visual_tool_does_not_expose_paths_or_bounding_boxes():
    visual_tool = next(tool for tool in DOCUMENT_TOOLS if tool.name == "analyze_document_visual")
    schema = visual_tool.tool_call_schema.model_json_schema()

    assert set(schema["properties"]) == {
        "source_file",
        "question",
        "visual_id",
        "page_number",
    }


def test_document_metadata_filters_decorative_and_duplicate_visuals():
    document = Document(
        file_name="front-page.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="logo",
                text="Company logo",
                type="image",
                block_type=BlockType.IMAGE,
                page_number=1,
                bbox=(0.0, 0.0, 100.0, 40.0),
            ),
            Block(
                id="photo",
                text="Figure 1: Factory photo",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                bbox=(20.0, 100.0, 300.0, 400.0),
            ),
            Block(
                id="photo-duplicate",
                text="Figure 1: Factory photo",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                bbox=(20.0, 100.0, 300.0, 400.0),
            ),
        ],
        metadata={"total_pages": 1},
    )

    _, llm_client = _invoke_single_read_tool(
        "get_document_metadata",
        {"source_file": "front-page.pdf", "page_number": 1},
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    statistics = payload["content_statistics"]
    assert statistics["raw_visual_count"] == 3
    assert statistics["content_visual_count"] == 1
    assert statistics["numbered_figure_count"] == 1
    assert statistics["numbered_figures"] == ["Figure 1"]


def test_tool_agent_discovers_and_analyzes_pdf_visual():
    pdf_bytes, document = make_visual_pdf()
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put(document.file_name, pdf_bytes)
    vlm_client = RecordingVLM()
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "report.pdf"},
                        "id": "outline-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "visual_id": "visual-revenue",
                            "question": "How did revenue change?",
                        },
                        "id": "visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Revenue rises from Q1 to Q2 [report.pdf, page 1]."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    tool_messages = tool_messages_seen_by_model(llm_client)
    outline = json.loads(str(tool_messages[0].content))
    analysis = json.loads(str(tool_messages[1].content))
    assert outline["visuals"][0]["visual_id"] == "visual-revenue"
    assert outline["visuals"][0]["description"] == "A bar chart with quarterly revenue."
    assert analysis["analysis"] == "Revenue rises from Q1 to Q2."
    assert analysis["page_number"] == 1
    assert analysis["visual_id"] == "visual-revenue"
    assert vlm_client.calls[0]["size"] == (488, 348)
    assert "How did revenue change?" in vlm_client.calls[0]["prompt"]
    assert "Quarterly revenue chart for 2026." in vlm_client.calls[0]["prompt"]
    assert state["response"].sources[0].chunk.id == "visual:report.pdf:visual-revenue"
    assert state["response"].sources[0].chunk.metadata["evidence_type"] == "visual_analysis"


def test_visual_tool_can_analyze_full_pdf_page_as_fallback():
    pdf_bytes, document = make_visual_pdf()
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put(document.file_name, pdf_bytes)
    vlm_client = RecordingVLM("The page contains a blue chart.")
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "page_number": 1,
                            "question": "What is visible on this page?",
                        },
                        "id": "page-visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The page contains a blue chart [report.pdf, page 1]."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["visual_id"] is None
    assert payload["page_number"] == 1
    assert vlm_client.calls[0]["size"] == (800, 600)
    assert state["response"].sources[0].chunk.id == "visual:report.pdf:page-1"


def test_visual_tool_obeys_configured_pixel_budget(monkeypatch):
    monkeypatch.setenv("VLM_MAX_VISUAL_PIXELS", "10000")
    pdf_bytes, document = make_visual_pdf()
    asset_store = InMemoryDocumentAssetStore()
    asset_store.put(document.file_name, pdf_bytes)
    vlm_client = RecordingVLM("The page contains a chart.")
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "page_number": 1,
                            "question": "What is visible?",
                        },
                        "id": "page-visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The page contains a chart [report.pdf, page 1]."),
        ]
    )

    invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        vlm_client=vlm_client,
        asset_store=asset_store,
    )

    width, height = vlm_client.calls[0]["size"]
    assert width * height <= 10000


def test_visual_tool_returns_configuration_error_to_agent():
    _, document = make_visual_pdf()
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "analyze_document_visual",
                        "args": {
                            "source_file": "report.pdf",
                            "page_number": 1,
                            "question": "Analyze the chart",
                        },
                        "id": "visual-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Visual analysis is unavailable."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
    )

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    assert "no VLM backend is configured" in str(tool_message.content)
    assert state["response"].sources == []
