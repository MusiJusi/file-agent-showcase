"""Runtime dependencies shared by read-only document tools."""

from dataclasses import dataclass

from file_agent.agent_tools.limits import DEFAULT_HISTORY_TURNS
from file_agent.document import Document
from file_agent.document_assets import DocumentAssetStore
from file_agent.llm.base import ToolCallingLLMClient
from file_agent.retrieval import Retriever
from file_agent.vlm.base import VLMClient


@dataclass(frozen=True)
class ToolAgentContext:
    llm_client: ToolCallingLLMClient
    retriever: Retriever
    documents: list[Document]
    default_top_k: int = 5
    max_tool_rounds: int = 4
    max_history_turns: int = DEFAULT_HISTORY_TURNS
    vlm_client: VLMClient | None = None
    asset_store: DocumentAssetStore | None = None
    require_evidence_tool: bool = False
    required_evidence_files: tuple[str, ...] = ()
    max_context_tokens: int = 90_000
