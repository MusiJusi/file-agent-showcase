"""Document discovery, outlines, and bounded content reads."""

import json
import re
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.agent_tools.common import (
    _block_matches_location,
    _block_page_number,
    _evidence_command,
    _find_document,
    _integer_metadata_values,
    _paginate_text,
    _render_blocks,
    _render_blocks_with_spans,
    _section_blocks,
    _source_metadata,
    _unique_metadata_values,
    _validate_location,
)
from file_agent.agent_tools.tables import _document_tables
from file_agent.agent_tools.visual import _document_visuals
from file_agent.document import Block, BlockType, Document


@tool
def list_documents(runtime: ToolRuntime[Any, dict]) -> str:
    """List uploaded documents, their structure, and useful source coordinates."""
    documents = []
    for document in runtime.context.documents:
        metadata = document.metadata
        sheets = _unique_metadata_values(document.blocks, "sheet_name")
        slides = _integer_metadata_values(document.blocks, "slide_number")
        documents.append(
            {
                "source_file": document.file_name,
                "file_type": document.file_type,
                "blocks": len(document.blocks),
                "total_pages": metadata.get("total_pages"),
                "total_slides": max(slides, default=None),
                "sheets": sheets,
                "headings": len(_document_outline(document)),
                "tables": len(_document_tables(document)),
                "visuals": len(_document_visuals(document)),
            }
        )
    return json.dumps({"documents": documents}, ensure_ascii=False, default=str)


@tool
def get_document_outline(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
) -> str:
    """Return readable section, location, table, and visual identifiers.

    Args:
        source_file: Exact file name returned by list_documents.
    """
    document = _find_document(runtime.context.documents, source_file)
    return json.dumps(
        {
            "source_file": document.file_name,
            "table_of_contents": _document_outline(document),
            "locations": _document_locations(document),
            "tables": _document_tables(document),
            "table_analysis_hint": (
                "Use analyze_table instead of paginating read_table for whole-table "
                "totals, distinct counts, extrema, or grouped sums."
            ),
            "visuals": _document_visuals(document),
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def read_document(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
    offset: int = 0,
) -> Command:
    """Read an uploaded document sequentially in bounded text pages.

    Use this for complete summaries, exhaustive extraction, or short/unstructured
    documents without a useful outline. For a topical question, prefer
    search_documents; for a large structured document, prefer section or location
    reads. Continue with the returned next_offset only when more content is needed.

    Args:
        source_file: Exact file name returned by list_documents.
        offset: Character offset returned as next_offset, or 0 for the first part.
    """
    document = _find_document(runtime.context.documents, source_file)
    content, block_spans = _render_blocks_with_spans(document.blocks)
    page = _paginate_text(content, offset)
    page_end = page["next_offset"] or len(content)
    page_blocks = [
        block
        for block, block_start, block_end in block_spans
        if block_end > offset and block_start < page_end
    ]
    metadata = _source_metadata(document, page_blocks)
    metadata.update(
        {
            "content_offset": offset,
            "evidence_type": "document_read",
        }
    )
    payload = {
        "source_file": document.file_name,
        **page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"document:{document.file_name}:{offset}",
        text=page["text"],
        metadata=metadata,
    )


@tool
def read_document_section(
    source_file: str,
    section_id: str,
    runtime: ToolRuntime[Any, dict],
    offset: int = 0,
) -> Command:
    """Read a section identified by get_document_outline, including subsections.

    Args:
        source_file: Exact file name returned by list_documents.
        section_id: Exact section_id returned by get_document_outline.
        offset: Character offset returned as next_offset, or 0 for the first part.
    """
    document = _find_document(runtime.context.documents, source_file)
    outline = _document_outline(document)
    normalized_id = section_id.strip()
    entry = next((item for item in outline if item.get("section_id") == normalized_id), None)
    if entry is None:
        raise ValueError(f"section is not indexed in {document.file_name}: {section_id}")

    blocks = _section_blocks(document, normalized_id, entry.get("level", 1))
    content = _render_blocks(blocks)
    page = _paginate_text(content, offset)
    metadata = _source_metadata(document, blocks)
    metadata.update(
        {
            "section": entry.get("title"),
            "section_id": normalized_id,
            "content_offset": offset,
        }
    )
    payload = {
        "source_file": document.file_name,
        "section_id": normalized_id,
        "title": entry.get("title"),
        **page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"section:{document.file_name}:{normalized_id}:{offset}",
        text=page["text"],
        metadata=metadata,
    )


@tool
def read_document_location(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
    page_number: int | None = None,
    slide_number: int | None = None,
    sheet_name: str | None = None,
    offset: int = 0,
) -> Command:
    """Read one page, slide, or sheet from an uploaded document.

    Exactly one location argument must be provided.

    Args:
        source_file: Exact file name returned by list_documents.
        page_number: One-indexed PDF or DOCX page number.
        slide_number: One-indexed PPTX slide number.
        sheet_name: Exact XLSX sheet name returned by list_documents.
        offset: Character offset returned as next_offset, or 0 for the first part.
    """
    document = _find_document(runtime.context.documents, source_file)
    locator_name, locator_value = _validate_location(page_number, slide_number, sheet_name)
    blocks = [
        block
        for block in document.blocks
        if _block_matches_location(block, locator_name, locator_value)
    ]
    if not blocks:
        raise ValueError(f"{locator_name}={locator_value!r} is not indexed in {document.file_name}")
    if locator_name == "sheet_name":
        locator_value = str(blocks[0].metadata["sheet_name"])

    content = _render_blocks(blocks)
    page = _paginate_text(content, offset)
    metadata = _source_metadata(document, blocks)
    metadata.update({locator_name: locator_value, "content_offset": offset})
    payload = {
        "source_file": document.file_name,
        "location": {locator_name: locator_value},
        **page,
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=(f"location:{document.file_name}:{locator_name}:{locator_value}:{offset}"),
        text=page["text"],
        metadata=metadata,
    )


def _document_outline(document: Document) -> list[dict[str, Any]]:
    raw_outline = document.metadata.get("table_of_contents") or []
    if not raw_outline:
        raw_outline = [
            {
                "title": block.text.strip(),
                "level": block.metadata.get("hierarchy_level", 1),
                "page": _block_page_number(block),
                "block_id": block.id,
            }
            for block in document.blocks
            if block.block_type == BlockType.HEADING and block.text.strip()
        ]

    outline = []
    for entry in raw_outline:
        block_id = entry.get("block_id") or entry.get("section_id")
        outline.append(
            {
                "section_id": block_id,
                "title": entry.get("title"),
                "level": entry.get("level", 1),
                "page_number": entry.get("page_number", entry.get("page")),
            }
        )
    return outline


def _document_locations(document: Document) -> dict[str, Any]:
    slides: list[dict[str, Any]] = []
    numbered_sections: list[dict[str, Any]] = []
    seen_slides: set[int] = set()
    for block in document.blocks:
        slide_number = block.metadata.get("slide_number")
        if (
            not isinstance(slide_number, int)
            or isinstance(slide_number, bool)
            or slide_number in seen_slides
        ):
            continue
        seen_slides.add(slide_number)
        title = _location_title(block.text, f"Slide {slide_number}")
        slides.append({"slide_number": slide_number, "title": title})
        section_number = _location_section_number(block.text, f"Slide {slide_number}")
        if section_number is not None:
            numbered_sections.append(
                {
                    "section_number": section_number,
                    "title": title,
                    "slide_number": slide_number,
                }
            )

    sheets = [
        {"sheet_name": sheet_name}
        for sheet_name in _unique_metadata_values(document.blocks, "sheet_name")
    ]
    locations = {"slides": slides, "sheets": sheets}
    if numbered_sections:
        locations["numbered_sections"] = numbered_sections
    section_review = _presentation_section_review(document, slides, numbered_sections)
    if section_review:
        locations["thematic_section_review"] = section_review
    return locations


def _presentation_section_review(
    document: Document,
    slides: list[dict[str, Any]],
    numbered_sections: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not numbered_sections:
        return None

    numbered_slides = {entry["slide_number"] for entry in numbered_sections}
    substantial_terms = (
        "barrier",
        "risk",
        "challenge",
        "limitation",
        "constraint",
        "барьер",
        "риск",
        "проблем",
        "ограничен",
    )
    unnumbered_candidates = [
        slide
        for slide in slides
        if slide["slide_number"] not in numbered_slides
        and isinstance(slide.get("title"), str)
        and any(term in slide["title"].casefold() for term in substantial_terms)
    ]
    first_numbered = numbered_sections[0]
    opening_candidates = [first_numbered] if first_numbered["slide_number"] <= 3 else []
    if not opening_candidates and not unnumbered_candidates:
        return None

    review: dict[str, Any] = {
        "opening_numbered_candidates": opening_candidates,
        "substantial_unnumbered_candidates": unnumbered_candidates,
        "guidance": (
            "Do not copy the numbered-divider count mechanically. Inspect opening "
            "candidates and exclude them when they are introductory; include each "
            "substantial unnumbered topic, then list the final thematic sections."
        ),
    }
    if opening_candidates and _opening_section_is_introductory(document, numbered_sections):
        review["introductory_numbered_sections"] = opening_candidates
        reconciled = [*numbered_sections[1:], *unnumbered_candidates]
        review["recommended_thematic_sections"] = sorted(
            (
                {
                    "title": entry["title"],
                    "slide_number": entry["slide_number"],
                }
                for entry in reconciled
            ),
            key=lambda entry: entry["slide_number"],
        )
    return review


def _opening_section_is_introductory(
    document: Document,
    numbered_sections: list[dict[str, Any]],
) -> bool:
    first_slide = numbered_sections[0]["slide_number"]
    next_slide = (
        numbered_sections[1]["slide_number"] if len(numbered_sections) > 1 else first_slide + 2
    )
    opening_text = "\n".join(
        block.text
        for block in document.blocks
        if isinstance((slide := block.metadata.get("slide_number")), int)
        and first_slide <= slide < next_slide
    ).casefold()
    introduction_terms = (
        "introduction",
        "introductory",
        "overview",
        "agenda",
        "this presentation",
        "presentation will",
        "введение",
        "вводн",
        "обзор",
        "повестк",
        "эта презентация",
        "презентация рассмотрит",
    )
    return any(term in opening_text for term in introduction_terms)


def _location_title(text: str, scaffold: str) -> str | None:
    for line in _location_content_lines(text, scaffold):
        return line.removeprefix("Title:").strip()[:300] or None
    return None


def _location_section_number(text: str, scaffold: str) -> int | None:
    lines = _location_content_lines(text, scaffold)
    if len(lines) != 2:
        return None
    match = re.fullmatch(r"(\d{1,3})[.)]?", lines[1])
    if match is None:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _location_content_lines(text: str, scaffold: str) -> list[str]:
    return [
        line
        for raw_line in text.splitlines()
        if (line := raw_line.strip()) and line != scaffold and line not in {"Text:", "Table:"}
    ]


def _document_outline_for_blocks(
    document: Document,
    blocks: list[Block],
) -> list[dict[str, Any]]:
    if len(blocks) == len(document.blocks):
        return _document_outline(document)
    block_ids = {block.id for block in blocks}
    pages = {_block_page_number(block) for block in blocks}
    return [
        entry
        for entry in _document_outline(document)
        if entry.get("section_id") in block_ids or entry.get("page_number") in pages
    ]
