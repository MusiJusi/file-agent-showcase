"""Source propagation, document selection, rendering, and pagination."""

import json
from typing import Any

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.agent_tools.limits import (
    MAX_TABLE_ROW_LENGTH,
    MAX_TABLE_ROWS,
    MAX_TOOL_CONTENT_LENGTH,
    MAX_TOOL_PASSAGE_LENGTH,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import SearchResult


def _find_document(documents: list[Document], source_file: str) -> Document:
    requested_name = source_file.strip().casefold()
    matches = [item for item in documents if item.file_name.casefold() == requested_name]
    if not matches:
        raise ValueError(f"document is not indexed: {source_file}")
    if len(matches) > 1:
        raise ValueError(f"multiple indexed documents share this file name: {source_file}")
    return matches[0]


def _section_blocks(document: Document, section_id: str, raw_level: Any) -> list[Block]:
    start = next(
        (index for index, block in enumerate(document.blocks) if block.id == section_id),
        None,
    )
    if start is None:
        raise ValueError(f"section block is not available in {document.file_name}: {section_id}")

    level = _heading_level(raw_level)
    end = len(document.blocks)
    for index in range(start + 1, len(document.blocks)):
        block = document.blocks[index]
        if block.block_type != BlockType.HEADING:
            continue
        if _heading_level(block.metadata.get("hierarchy_level", 1)) <= level:
            end = index
            break
    return document.blocks[start:end]


def _heading_level(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _render_blocks(blocks: list[Block]) -> str:
    text, _ = _render_blocks_with_spans(blocks)
    return text


def _render_blocks_with_spans(
    blocks: list[Block],
) -> tuple[str, list[tuple[Block, int, int]]]:
    parts: list[str] = []
    spans: list[tuple[Block, int, int]] = []
    cursor = 0
    for block in blocks:
        rendered = block.to_markdown().strip()
        if not rendered:
            continue
        if parts:
            cursor += 2
        start = cursor
        parts.append(rendered)
        cursor += len(rendered)
        spans.append((block, start, cursor))
    text = "\n\n".join(parts)
    if not text:
        raise ValueError("the selected document part contains no readable text")
    return text, spans


def _validate_location(
    page_number: int | None,
    slide_number: int | None,
    sheet_name: str | None,
) -> tuple[str, int | str]:
    provided = [
        ("page_number", page_number) if page_number is not None else None,
        ("slide_number", slide_number) if slide_number is not None else None,
        ("sheet_name", sheet_name) if sheet_name is not None else None,
    ]
    selected = [item for item in provided if item is not None]
    if len(selected) != 1:
        raise ValueError("provide exactly one of page_number, slide_number, or sheet_name")

    name, value = selected[0]
    if name in ("page_number", "slide_number"):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return name, value

    normalized_sheet = str(value).strip()
    if not normalized_sheet:
        raise ValueError("sheet_name must not be empty")
    return name, normalized_sheet


def _block_matches_location(block: Block, name: str, value: int | str) -> bool:
    if name == "page_number":
        return _block_page_number(block) == value
    if name == "slide_number":
        return block.metadata.get("slide_number") == value
    block_sheet = block.metadata.get("sheet_name")
    return isinstance(block_sheet, str) and block_sheet.casefold() == str(value).casefold()


def _block_page_number(block: Block) -> int | None:
    if block.page_number is not None:
        return block.page_number
    value = block.metadata.get("page_number")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _source_metadata(document: Document, blocks: list[Block]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_file": document.file_name,
        "file_type": document.file_type,
        "block_ids": [block.id for block in blocks],
    }
    pages = sorted({page for block in blocks if (page := _block_page_number(block)) is not None})
    slides = _integer_metadata_values(blocks, "slide_number")
    sheets = _unique_metadata_values(blocks, "sheet_name")
    dataset_doc_ids = _unique_metadata_values(blocks, "dataset_doc_id")
    dataset_record_ids = _unique_metadata_values(blocks, "dataset_record_id")
    if pages:
        metadata["page_number"] = pages[0]
        metadata["page_numbers"] = pages
    if slides:
        metadata["slide_number"] = slides[0]
        metadata["slide_numbers"] = slides
    if sheets:
        metadata["sheet_name"] = sheets[0]
        metadata["sheet_names"] = sheets
    if len(dataset_doc_ids) == 1:
        metadata["dataset_doc_id"] = dataset_doc_ids[0]
    if len(dataset_record_ids) == 1:
        metadata["dataset_record_id"] = dataset_record_ids[0]
    return metadata


def _document_level_source_metadata(document: Document) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_file": document.file_name,
        "file_type": document.file_type,
    }
    dataset_doc_ids = _unique_metadata_values(document.blocks, "dataset_doc_id")
    dataset_record_ids = _unique_metadata_values(document.blocks, "dataset_record_id")
    if len(dataset_doc_ids) == 1:
        metadata["dataset_doc_id"] = dataset_doc_ids[0]
    if len(dataset_record_ids) == 1:
        metadata["dataset_record_id"] = dataset_record_ids[0]
    return metadata


def _document_total_pages(document: Document) -> int:
    configured = _non_negative_metadata_int(document.metadata.get("total_pages"))
    if configured is not None:
        return configured
    return max(
        (page for block in document.blocks if (page := _block_page_number(block)) is not None),
        default=0,
    )


def _non_negative_metadata_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _original_block_text(block: Block) -> str:
    """Return source-extracted text without generated visual descriptions."""
    text = block.text or ""
    if block.block_type in (BlockType.FIGURE, BlockType.IMAGE) and block.vlm_description:
        generated = f"[Image description]: {block.vlm_description}"
        position = text.rfind(generated)
        if position >= 0 and not text[position + len(generated) :].strip():
            text = text[:position].rstrip()
    return text


def _blocks_extracted_text(blocks: list[Block]) -> str:
    return "\n".join(text for block in blocks if (text := _original_block_text(block).strip()))


def _integer_metadata_values(blocks: list[Block], key: str) -> list[int]:
    return sorted(
        {
            value
            for block in blocks
            if isinstance((value := block.metadata.get(key)), int) and not isinstance(value, bool)
        }
    )


def _unique_metadata_values(blocks: list[Block], key: str) -> list[str]:
    values: list[str] = []
    for block in blocks:
        value = block.metadata.get(key)
        if isinstance(value, str) and value not in values:
            values.append(value)
    return values


def _paginate_text(text: str, offset: int) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if offset >= len(text) and text:
        raise ValueError("offset is beyond the available content")

    end = min(len(text), offset + MAX_TOOL_CONTENT_LENGTH)
    if end < len(text):
        boundary = max(text.rfind("\n", offset, end), text.rfind(" ", offset, end))
        if boundary > offset + MAX_TOOL_CONTENT_LENGTH // 2:
            end = boundary
    page = text[offset:end].strip()
    return {
        "text": page,
        "offset": offset,
        "next_offset": end if end < len(text) else None,
        "total_characters": len(text),
    }


def _paginate_rows(rows: list[str], offset: int, limit: int) -> dict[str, Any]:
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if not rows:
        raise ValueError("the selected table contains no readable rows")
    if offset >= len(rows):
        raise ValueError("offset is beyond the available table rows")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")

    bounded_limit = max(1, min(limit, MAX_TABLE_ROWS))
    selected: list[str] = []
    used_characters = 0
    for row in rows[offset : offset + bounded_limit]:
        bounded_row = row
        if len(bounded_row) > MAX_TABLE_ROW_LENGTH:
            bounded_row = bounded_row[:MAX_TABLE_ROW_LENGTH].rstrip() + "..."
        addition = len(bounded_row) + (1 if selected else 0)
        if selected and used_characters + addition > MAX_TOOL_CONTENT_LENGTH:
            break
        selected.append(bounded_row)
        used_characters += addition

    next_offset = offset + len(selected)
    return {
        "rows": selected,
        "offset": offset,
        "next_offset": next_offset if next_offset < len(rows) else None,
        "total_rows": len(rows),
    }


def _tool_command(
    payload: dict[str, Any],
    runtime: ToolRuntime[Any, dict],
    sources: list[SearchResult] | None = None,
    search_query: str | None = None,
) -> Command:
    update: dict[str, Any] = {
        "messages": [
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                tool_call_id=runtime.tool_call_id,
            )
        ]
    }
    if sources:
        update["sources"] = sources
    if search_query is not None:
        update["search_queries"] = [search_query]
    return Command(update=update)


def _evidence_command(
    payload: dict[str, Any],
    runtime: ToolRuntime[Any, dict],
    evidence_id: str,
    text: str,
    metadata: dict[str, Any],
) -> Command:
    source = SearchResult(
        chunk=Chunk(id=evidence_id, text=text, metadata=metadata),
        score=1.0,
    )
    return _tool_command(payload, runtime, sources=[source])


def _truncate_passage(passage: str) -> str:
    if len(passage) <= MAX_TOOL_PASSAGE_LENGTH:
        return passage
    return passage[:MAX_TOOL_PASSAGE_LENGTH].rstrip() + "..."
