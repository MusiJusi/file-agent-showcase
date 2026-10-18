"""Read-only agent tools, grouped by responsibility.

The public imports and tool registration order are preserved for callers.
"""

from langchain_core.tools import BaseTool

from file_agent.agent_tools.context import ToolAgentContext
from file_agent.agent_tools.limits import (
    DEFAULT_HISTORY_TURNS,
    LLM_CONTEXT_METADATA_KEY,
    MAX_CALCULATION_VALUES,
    MAX_OCCURRENCE_CONTEXT_LENGTH,
    MAX_OCCURRENCE_EXAMPLES,
    MAX_OCCURRENCE_QUERY_LENGTH,
    MAX_TABLE_ANALYSIS_RESULTS,
    MAX_TABLE_LOOKUP_RESULTS,
    MAX_TABLE_ROW_LENGTH,
    MAX_TABLE_ROWS,
    MAX_TOOL_CONTENT_LENGTH,
    MAX_TOOL_PASSAGE_LENGTH,
    MAX_TOOL_QUERY_LENGTH,
    MAX_TOOL_TOP_K,
    MAX_VISUAL_CONTEXT_LENGTH,
    MAX_VISUAL_DESCRIPTION_LENGTH,
    MAX_VISUAL_PIXELS,
    MAX_VISUAL_QUESTION_LENGTH,
    VISUAL_CROP_PADDING,
    WORD_PATTERN,
)
from file_agent.agent_tools.navigation import (
    get_document_outline,
    list_documents,
    read_document,
    read_document_location,
    read_document_section,
)
from file_agent.agent_tools.prompts import (
    TOOL_AGENT_SYSTEM_PROMPT,
    TOOL_LIMIT_MESSAGE,
    VISUAL_ANALYSIS_PROMPT,
)
from file_agent.agent_tools.search import read_source_context, search_documents
from file_agent.agent_tools.statistics import count_document_occurrences, get_document_metadata
from file_agent.agent_tools.tables import (
    analyze_table,
    calculate,
    compare_table_columns,
    lookup_table_rows,
    read_table,
)
from file_agent.agent_tools.visual import analyze_document_visual

DOCUMENT_TOOLS: list[BaseTool] = [
    search_documents,
    read_source_context,
    list_documents,
    get_document_metadata,
    count_document_occurrences,
    get_document_outline,
    read_document,
    read_document_section,
    read_document_location,
    analyze_table,
    lookup_table_rows,
    compare_table_columns,
    read_table,
    calculate,
    analyze_document_visual,
]

__all__ = [
    "DEFAULT_HISTORY_TURNS",
    "DOCUMENT_TOOLS",
    "LLM_CONTEXT_METADATA_KEY",
    "MAX_CALCULATION_VALUES",
    "MAX_OCCURRENCE_CONTEXT_LENGTH",
    "MAX_OCCURRENCE_EXAMPLES",
    "MAX_OCCURRENCE_QUERY_LENGTH",
    "MAX_TABLE_ANALYSIS_RESULTS",
    "MAX_TABLE_LOOKUP_RESULTS",
    "MAX_TABLE_ROWS",
    "MAX_TABLE_ROW_LENGTH",
    "MAX_TOOL_CONTENT_LENGTH",
    "MAX_TOOL_PASSAGE_LENGTH",
    "MAX_TOOL_QUERY_LENGTH",
    "MAX_TOOL_TOP_K",
    "MAX_VISUAL_CONTEXT_LENGTH",
    "MAX_VISUAL_DESCRIPTION_LENGTH",
    "MAX_VISUAL_PIXELS",
    "MAX_VISUAL_QUESTION_LENGTH",
    "TOOL_AGENT_SYSTEM_PROMPT",
    "TOOL_LIMIT_MESSAGE",
    "ToolAgentContext",
    "VISUAL_ANALYSIS_PROMPT",
    "VISUAL_CROP_PADDING",
    "WORD_PATTERN",
    "analyze_document_visual",
    "analyze_table",
    "calculate",
    "compare_table_columns",
    "count_document_occurrences",
    "get_document_metadata",
    "get_document_outline",
    "list_documents",
    "lookup_table_rows",
    "read_document",
    "read_document_location",
    "read_document_section",
    "read_source_context",
    "read_table",
    "search_documents",
]
