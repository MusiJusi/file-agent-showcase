"""Support."""

import fitz
from langchain_core.messages import AIMessage, ToolMessage
from PIL import Image

from file_agent.agent_tools import ToolAgentContext
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.rag_graph import build_tool_agent_graph
from file_agent.retrieval import SearchResult
from file_agent.vlm.base import VLMClient


class FakeToolCallingLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt: str) -> str:
        raise AssertionError("generate must not be used by the tool agent")

    def chat_with_tools(self, messages, tools, tool_choice="auto"):
        self.calls.append(
            {
                "messages": list(messages),
                "tool_names": [tool.name for tool in tools],
                "tool_choice": tool_choice,
            }
        )
        if not self.responses:
            raise AssertionError("Unexpected tool-calling LLM invocation")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class FakeRetriever:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.search_calls = []

    def index(self, chunks):
        self.results = [SearchResult(chunk=chunk, score=1.0) for chunk in chunks]

    def search(self, query: str, top_k: int = 5, source_file: str | None = None):
        self.search_calls.append((query, top_k, source_file))
        results = self.results
        if source_file is not None:
            results = [
                result
                for result in results
                if result.chunk.metadata.get("source_file") == source_file
            ]
        return results[:top_k]

    def clear(self):
        self.results = []


def make_document() -> Document:
    return Document(
        file_name="plan.md",
        file_type=".md",
        blocks=[
            Block(
                id="block-1",
                text="The project deadline is Friday.",
                type="text",
                metadata={"source_file": "plan.md"},
            )
        ],
        metadata={
            "table_of_contents": [{"title": "Schedule", "level": 1, "block_id": "heading-1"}]
        },
    )


def make_search_result() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id="chunk-1",
            text="deadline fragment",
            metadata={
                "context": "The project deadline is Friday.",
                "source_file": "plan.md",
                "page_number": 2,
            },
        ),
        score=0.9,
    )


def invoke_tool_agent(
    llm_client,
    retriever,
    max_tool_rounds=4,
    documents=None,
    vlm_client=None,
    asset_store=None,
    require_evidence_tool=False,
    required_evidence_files=(),
    question="When is the deadline?",
    max_context_tokens=90_000,
):
    return build_tool_agent_graph().invoke(
        {
            "question": question,
            "documents_count": 1,
            "chunks_count": 1,
        },
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=documents or [make_document()],
            max_tool_rounds=max_tool_rounds,
            vlm_client=vlm_client,
            asset_store=asset_store,
            require_evidence_tool=require_evidence_tool,
            required_evidence_files=required_evidence_files,
            max_context_tokens=max_context_tokens,
        ),
    )


def tool_messages_seen_by_model(llm_client):
    return [
        message for message in llm_client.calls[-1]["messages"] if isinstance(message, ToolMessage)
    ]


def make_structured_document() -> Document:
    return Document(
        file_name="handbook.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="heading-main",
                text="Main section",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=1,
            ),
            Block(
                id="main-body",
                text="Main section body.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="heading-child",
                text="Child section",
                type="heading",
                metadata={"hierarchy_level": 2},
                block_type=BlockType.HEADING,
                page_number=2,
            ),
            Block(
                id="child-body",
                text="Child section body.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=2,
            ),
            Block(
                id="table-1",
                text="| Name | Value |\n| --- | --- |\n| Accuracy | 0.9 |",
                type="table",
                block_type=BlockType.TABLE,
                page_number=2,
            ),
            Block(
                id="heading-next",
                text="Next section",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=3,
            ),
            Block(
                id="next-body",
                text="This text must not be returned with the main section.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=3,
            ),
        ],
    )


def _invoke_single_read_tool(tool_name, arguments, documents):
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": arguments,
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Answer based on the selected location."),
        ]
    )
    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=documents,
    )
    return state, llm_client


def make_metadata_document() -> Document:
    description = "Alpha chart generated by the VLM"
    return Document(
        file_name="annual-report.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="heading-1",
                text="Annual Report",
                type="heading",
                metadata={"hierarchy_level": 1, "dataset_doc_id": "docbench/0.pdf"},
                block_type=BlockType.HEADING,
                page_number=1,
            ),
            Block(
                id="body-1",
                text="Alpha beta state-of-the-art. Alpha.",
                type="text",
                metadata={"dataset_doc_id": "docbench/0.pdf"},
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="visual-1",
                text=f"Original caption\n\n[Image description]: {description}",
                type="figure",
                metadata={"dataset_doc_id": "docbench/0.pdf"},
                block_type=BlockType.FIGURE,
                page_number=2,
                bbox=(10.0, 10.0, 100.0, 100.0),
                vlm_description=description,
            ),
        ],
        metadata={
            "total_pages": 4,
            "file_size_bytes": 12345,
            "native_pdf_word_counts_by_page": [6, 0, 0, 0],
            "native_pdf_word_count": 6,
            "parsing_method": "docling",
            "pdf_metadata": {"title": "2026 Annual Report", "author": "Ada Lovelace"},
        },
    )


def make_occurrence_document() -> Document:
    description = "memory memory"
    return Document(
        file_name="memory.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="page-1",
                text="Agentic memory supports memory. MEMORY. memoryless.",
                type="text",
                metadata={"dataset_doc_id": "docbench/memory.pdf"},
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="page-2",
                text="A memory-based method describes agentic\nmemory systems.",
                type="text",
                metadata={"dataset_doc_id": "docbench/memory.pdf"},
                block_type=BlockType.TEXT,
                page_number=2,
            ),
            Block(
                id="visual-3",
                text=f"[Image description]: {description}",
                type="figure",
                metadata={"dataset_doc_id": "docbench/memory.pdf"},
                block_type=BlockType.FIGURE,
                page_number=3,
                vlm_description=description,
            ),
        ],
        metadata={"total_pages": 3},
    )


class RecordingVLM(VLMClient):
    def __init__(self, answer="Revenue rises from Q1 to Q2."):
        self.answer = answer
        self.calls = []

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        self.calls.append({"size": image.size, "prompt": prompt})
        return self.answer


def make_visual_pdf() -> tuple[bytes, Document]:
    pdf = fitz.open()
    page = pdf.new_page(width=400, height=300)
    page.insert_text((30, 30), "Quarterly revenue")
    page.draw_rect(fitz.Rect(80, 80, 300, 230), fill=(0, 0, 1))
    contents = pdf.tobytes()
    pdf.close()

    document = Document(
        file_name="report.pdf",
        file_type="pdf",
        blocks=[
            Block(
                id="heading-revenue",
                text="Revenue",
                type="heading",
                metadata={"hierarchy_level": 1},
                block_type=BlockType.HEADING,
                page_number=1,
            ),
            Block(
                id="text-revenue",
                text="Quarterly revenue chart for 2026.",
                type="text",
                block_type=BlockType.TEXT,
                page_number=1,
            ),
            Block(
                id="visual-revenue",
                text="",
                type="figure",
                block_type=BlockType.FIGURE,
                page_number=1,
                bbox=(80.0, 80.0, 300.0, 230.0),
                vlm_description="A bar chart with quarterly revenue.",
            ),
        ],
    )
    document.build_table_of_contents()
    return contents, document


def invoke_persistent_turn(
    graph,
    llm_client,
    retriever,
    question,
    thread_id="conversation-1",
    max_history_turns=6,
):
    return graph.invoke(
        {
            "question": question,
            "documents_count": 1,
            "chunks_count": 1,
        },
        config={"configurable": {"thread_id": thread_id}},
        context=ToolAgentContext(
            llm_client=llm_client,
            retriever=retriever,
            documents=[make_document()],
            max_history_turns=max_history_turns,
        ),
    )
