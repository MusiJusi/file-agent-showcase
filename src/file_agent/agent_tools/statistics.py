"""Exact document metadata, content statistics, and occurrence counts."""

import hashlib
import json
import re
from typing import Any, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.agent_tools.common import (
    _block_page_number,
    _blocks_extracted_text,
    _document_level_source_metadata,
    _document_total_pages,
    _evidence_command,
    _find_document,
    _heading_level,
    _integer_metadata_values,
    _non_negative_metadata_int,
    _original_block_text,
    _unique_metadata_values,
)
from file_agent.agent_tools.limits import (
    MAX_OCCURRENCE_CONTEXT_LENGTH,
    MAX_OCCURRENCE_EXAMPLES,
    MAX_OCCURRENCE_QUERY_LENGTH,
    WORD_PATTERN,
)
from file_agent.agent_tools.navigation import _document_outline_for_blocks
from file_agent.agent_tools.tables import _document_tables
from file_agent.agent_tools.visual import _document_visuals
from file_agent.document import Block, BlockType, Document


@tool
def get_document_metadata(
    source_file: str,
    runtime: ToolRuntime[Any, dict],
    page_number: int | None = None,
) -> Command:
    """Return exact document properties, structure counts, and text statistics.

    Use this for document-level questions about PDF page count, file size, title,
    author, creation metadata, or word/character counts. Word counts prefer the
    native PDF text layer and fall back to parsed/OCR text on image-only pages.
    Statistics cover the complete document or one requested page and exclude
    VLM-generated image descriptions.

    Args:
        source_file: Exact file name returned by list_documents.
        page_number: Optional one-indexed PDF/DOCX page for page-level text counts.
    """
    document = _find_document(runtime.context.documents, source_file)
    total_pages = _document_total_pages(document)
    if page_number is not None:
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
            raise ValueError("page_number must be a positive integer")
        if total_pages and page_number > total_pages:
            raise ValueError(f"page_number exceeds document page count: {total_pages}")
        statistic_blocks = [
            block for block in document.blocks if _block_page_number(block) == page_number
        ]
    else:
        statistic_blocks = document.blocks
    text = _blocks_extracted_text(statistic_blocks)
    word_statistics = _word_statistics(document, statistic_blocks, page_number)
    pages = sorted(
        {page for block in statistic_blocks if (page := _block_page_number(block)) is not None}
    )
    slides = _integer_metadata_values(document.blocks, "slide_number")
    sheets = _unique_metadata_values(document.blocks, "sheet_name")
    pdf_metadata = document.metadata.get("pdf_metadata")
    if not isinstance(pdf_metadata, dict):
        pdf_metadata = {}

    scoped_outline = _document_outline_for_blocks(document, statistic_blocks)
    scoped_tables = [
        table
        for table in _document_tables(document)
        if page_number is None or table.get("page_number") == page_number
    ]
    scoped_visuals = [
        visual
        for visual in _document_visuals(document)
        if page_number is None or visual.get("page_number") == page_number
    ]
    content_statistics = _document_content_statistics(document, statistic_blocks)
    payload = {
        "source_file": document.file_name,
        "file_type": document.file_type,
        "file_size_bytes": _non_negative_metadata_int(document.metadata.get("file_size_bytes")),
        "total_pages": total_pages,
        "page_numbering": "1-indexed physical PDF/DOCX pages",
        "total_slides": max(slides, default=None),
        "sheets": sheets,
        "structure": {
            "scope": "full_document" if page_number is None else "page",
            "page_number": page_number,
            "blocks": len(statistic_blocks),
            "non_empty_blocks": sum(
                1 for block in statistic_blocks if _original_block_text(block).strip()
            ),
            "headings": len(scoped_outline),
            "top_level_sections": content_statistics["top_level_sections"],
            "tables": len(scoped_tables),
            "visuals": len(scoped_visuals),
            "content_visuals": content_statistics["content_visual_count"],
            "numbered_figures": content_statistics["numbered_figure_count"],
            "pages_with_extracted_blocks": pages,
        },
        "content_statistics": content_statistics,
        "text_statistics": {
            "scope": "full_document" if page_number is None else "page",
            "page_number": page_number,
            **word_statistics,
            "character_count_with_spaces": len(text),
            "character_count_without_whitespace": len(re.sub(r"\s", "", text)),
            "non_empty_line_count": sum(1 for line in text.splitlines() if line.strip()),
            "counting_method": (
                "native PDF text-layer words when available; Unicode parsed/OCR "
                "word fallback on zero-text pages; VLM image descriptions excluded"
            ),
        },
        "pdf_metadata": {
            str(key): value
            for key, value in pdf_metadata.items()
            if value not in (None, "", [], {})
        },
        "processing": {
            "parsing_method": document.metadata.get("parsing_method"),
            "ocr_engine": document.metadata.get("ocr_engine"),
            "page_analysis": document.metadata.get("page_analysis"),
            "vlm_described_figures": document.metadata.get("vlm_described_figures", 0),
        },
    }
    source_metadata = _document_level_source_metadata(document)
    source_metadata.update({"evidence_type": "document_metadata"})
    if page_number is not None:
        source_metadata["page_number"] = page_number
        source_metadata["page_numbers"] = [page_number]
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"metadata:{document.file_name}",
        text=evidence_text,
        metadata=source_metadata,
    )


@tool
def count_document_occurrences(
    source_file: str,
    query: str,
    runtime: ToolRuntime[Any, dict],
    match_mode: Literal["whole_word", "substring"] = "whole_word",
    case_sensitive: bool = False,
    page_number: int | None = None,
    max_examples: int = 10,
) -> Command:
    """Exhaustively count a literal term or phrase in an extracted document.

    Unlike semantic retrieval, this scans every parsed block and returns the exact
    non-overlapping count, all matched PDF/DOCX pages, and bounded context examples.
    VLM-generated image descriptions are excluded so they cannot inflate counts.

    Args:
        source_file: Exact file name returned by list_documents.
        query: Literal term or phrase to count, up to 500 characters.
        match_mode: whole_word for standalone word/phrase matches; substring for
            matches inside longer words or exact character sequences.
        case_sensitive: Whether letter case must match exactly.
        page_number: Optional one-indexed PDF/DOCX page to restrict the count to.
        max_examples: Number of matching contexts to return, from 1 to 20.
    """
    document = _find_document(runtime.context.documents, source_file)
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if len(normalized_query) > MAX_OCCURRENCE_QUERY_LENGTH:
        raise ValueError(f"query must not exceed {MAX_OCCURRENCE_QUERY_LENGTH} characters")
    if not isinstance(case_sensitive, bool):
        raise ValueError("case_sensitive must be a boolean")
    if page_number is not None:
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
            raise ValueError("page_number must be a positive integer")
        total_pages = _document_total_pages(document)
        if total_pages and page_number > total_pages:
            raise ValueError(f"page_number exceeds document page count: {total_pages}")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool):
        raise ValueError("max_examples must be an integer")
    bounded_examples = max(1, min(max_examples, MAX_OCCURRENCE_EXAMPLES))
    pattern = _occurrence_pattern(normalized_query, match_mode, case_sensitive)

    total_occurrences = 0
    matched_blocks: list[Block] = []
    matched_block_ids: set[str] = set()
    matched_pages: set[int] = set()
    examples: list[dict[str, Any]] = []
    searched_blocks = 0
    for block in document.blocks:
        block_page = _block_page_number(block)
        if page_number is not None and block_page != page_number:
            continue
        text = _original_block_text(block)
        if not text:
            continue
        searched_blocks += 1
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        if block.id not in matched_block_ids:
            matched_block_ids.add(block.id)
            matched_blocks.append(block)
        if block_page is not None:
            matched_pages.add(block_page)
        total_occurrences += len(matches)
        for match in matches:
            if len(examples) >= bounded_examples:
                break
            examples.append(
                {
                    "block_id": block.id,
                    "page_number": block_page,
                    "matched_text": match.group(0),
                    "context": _match_context(text, match.start(), match.end()),
                }
            )

    payload = {
        "source_file": document.file_name,
        "query": normalized_query,
        "match_mode": match_mode,
        "case_sensitive": case_sensitive,
        "page_number_filter": page_number,
        "total_occurrences": total_occurrences,
        "matched_blocks": len(matched_blocks),
        "matched_pages": sorted(matched_pages),
        "searched_blocks": searched_blocks,
        "examples": examples,
        "examples_truncated": total_occurrences > len(examples),
        "counting_method": (
            "complete parsed document; non-overlapping literal matches; flexible "
            "whitespace for whole_word phrases; VLM descriptions excluded"
        ),
    }
    source_metadata = _document_level_source_metadata(document)
    source_metadata.update(
        {
            "evidence_type": "document_occurrence_count",
            "query": normalized_query,
            "match_mode": match_mode,
            "case_sensitive": case_sensitive,
            "total_occurrences": total_occurrences,
        }
    )
    if page_number is not None:
        source_metadata["page_number"] = page_number
        source_metadata["page_numbers"] = [page_number]
    elif matched_pages:
        source_metadata["page_number"] = min(matched_pages)
        source_metadata["page_numbers"] = sorted(matched_pages)
    digest = hashlib.sha256(
        f"{document.file_name}\0{normalized_query}\0{match_mode}\0"
        f"{case_sensitive}\0{page_number}".encode()
    ).hexdigest()[:12]
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"occurrences:{document.file_name}:{digest}",
        text=evidence_text,
        metadata=source_metadata,
    )


def _document_content_statistics(
    document: Document,
    blocks: list[Block],
) -> dict[str, Any]:
    outline = _document_outline_for_blocks(document, blocks)
    levels = [_heading_level(entry.get("level")) for entry in outline]
    top_level = min(levels, default=1)
    top_sections = [
        {"title": entry.get("title"), "page_number": entry.get("page_number")}
        for entry in outline
        if _heading_level(entry.get("level")) == top_level
    ]

    block_ids = {block.id for block in blocks}
    visuals = [visual for visual in _document_visuals(document) if visual["visual_id"] in block_ids]
    decorative_terms = (
        "logo",
        "masthead",
        "nameplate",
        "brand mark",
        "decorative icon",
        "логотип",
        "эмблема",
    )
    content_visuals = []
    seen_visual_keys: set[tuple[Any, ...]] = set()
    for visual in visuals:
        if any(
            term in str(visual.get("description") or "").casefold() for term in decorative_terms
        ):
            continue
        bbox = visual.get("bbox")
        rounded_bbox = (
            tuple(round(float(value), 1) for value in bbox)
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4
            else None
        )
        visual_key = (
            visual.get("page_number"),
            rounded_bbox,
            " ".join(str(visual.get("description") or "").casefold().split()),
        )
        if visual_key in seen_visual_keys:
            continue
        seen_visual_keys.add(visual_key)
        content_visuals.append(visual)

    numbered_figure_pattern = re.compile(
        r"\b(?:figure|fig\.?|рис(?:унок|\.)?)\s*([A-ZА-ЯЁ]?\d+(?:[.\-]\d+)*)",
        flags=re.IGNORECASE,
    )
    numbered_figures: dict[str, str] = {}
    for block in blocks:
        for match in numbered_figure_pattern.finditer(_original_block_text(block)):
            numbered_figures.setdefault(match.group(1).casefold(), match.group(0))

    text_blocks = [
        block
        for block in blocks
        if block.block_type
        not in (BlockType.HEADING, BlockType.TABLE, BlockType.FIGURE, BlockType.IMAGE)
        and _original_block_text(block).strip()
    ]
    first_text = _original_block_text(text_blocks[0]).strip() if text_blocks else ""
    sentence_match = re.search(r".+?(?:[.!?](?:\s|$)|$)", first_text, flags=re.DOTALL)
    first_sentence = " ".join(sentence_match.group(0).split()) if sentence_match else None

    abbreviation_pattern = re.compile(
        r"(?<![\w-])(?:[A-ZА-ЯЁ]{2,10})(?:-[A-ZА-ЯЁ]{2,10})?(?![\w-])"
    )
    abbreviations = sorted(
        {
            match.group(0)
            for block in blocks
            for match in abbreviation_pattern.finditer(_original_block_text(block))
        },
        key=str.casefold,
    )
    reference_entries, footnotes = _count_reference_and_footnote_entries(blocks)
    return {
        "top_level_section_count": len(top_sections),
        "top_level_sections": top_sections,
        "raw_visual_count": len(visuals),
        "content_visual_count": len(content_visuals),
        "content_visual_ids": [visual["visual_id"] for visual in content_visuals],
        "numbered_figure_count": len(numbered_figures),
        "numbered_figures": list(numbered_figures.values()),
        "reference_entry_count": reference_entries,
        "footnote_count": footnotes,
        "abbreviation_count": len(abbreviations),
        "abbreviations": abbreviations[:50],
        "abbreviations_truncated": len(abbreviations) > 50,
        "first_content_sentence": first_sentence,
    }


def _count_reference_and_footnote_entries(blocks: list[Block]) -> tuple[int, int]:
    reference_heading = re.compile(
        r"^(?:references|bibliography|works cited|литература|список литературы)$",
        flags=re.IGNORECASE,
    )
    entry_pattern = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+\S")
    in_references = False
    reference_level = 1
    references = 0
    footnotes = 0
    for block in blocks:
        text = _original_block_text(block).strip()
        if block.block_type == BlockType.HEADING:
            level = _heading_level(block.metadata.get("hierarchy_level"))
            if reference_heading.fullmatch(text):
                in_references = True
                reference_level = level
                continue
            if in_references and level <= reference_level:
                in_references = False
        if in_references:
            references += sum(1 for line in text.splitlines() if entry_pattern.match(line))
        role = str(block.metadata.get("label") or block.metadata.get("role") or "").casefold()
        if "footnote" in role or "foot_note" in role:
            footnotes += 1
    return references, footnotes


def _word_statistics(
    document: Document,
    blocks: list[Block],
    page_number: int | None,
) -> dict[str, Any]:
    extracted_word_count = sum(
        len(WORD_PATTERN.findall(text))
        for block in blocks
        if (text := _original_block_text(block).strip())
    )
    raw_native_counts = document.metadata.get("native_pdf_word_counts_by_page")
    if not (
        isinstance(raw_native_counts, list)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in raw_native_counts
        )
    ):
        return {
            "word_count": extracted_word_count,
            "extracted_word_count": extracted_word_count,
            "native_pdf_text_layer_word_count": None,
            "word_count_source": "parsed_document",
            "native_zero_text_pages_using_parsed_fallback": [],
        }

    native_counts: list[int] = raw_native_counts
    if page_number is not None:
        native_count = native_counts[page_number - 1] if page_number <= len(native_counts) else 0
        use_extracted = native_count == 0 and extracted_word_count > 0
        return {
            "word_count": extracted_word_count if use_extracted else native_count,
            "extracted_word_count": extracted_word_count,
            "native_pdf_text_layer_word_count": native_count,
            "word_count_source": (
                "parsed_document_fallback" if use_extracted else "native_pdf_text_layer"
            ),
            "native_zero_text_pages_using_parsed_fallback": (
                [page_number] if use_extracted else []
            ),
        }

    extracted_by_page: dict[int, int] = {}
    unpaged_word_count = 0
    for block in blocks:
        count = len(WORD_PATTERN.findall(_original_block_text(block)))
        block_page = _block_page_number(block)
        if block_page is None:
            unpaged_word_count += count
        else:
            extracted_by_page[block_page] = extracted_by_page.get(block_page, 0) + count

    effective_count = unpaged_word_count
    fallback_pages: list[int] = []
    last_page = max(len(native_counts), max(extracted_by_page, default=0))
    for current_page in range(1, last_page + 1):
        native_count = native_counts[current_page - 1] if current_page <= len(native_counts) else 0
        extracted_count = extracted_by_page.get(current_page, 0)
        if native_count > 0 or extracted_count == 0:
            effective_count += native_count
        else:
            effective_count += extracted_count
            fallback_pages.append(current_page)

    return {
        "word_count": effective_count,
        "extracted_word_count": extracted_word_count,
        "native_pdf_text_layer_word_count": sum(native_counts),
        "word_count_source": "native_pdf_text_layer_with_parsed_fallback",
        "native_zero_text_pages_using_parsed_fallback": fallback_pages,
    }


def _occurrence_pattern(
    query: str,
    match_mode: Literal["whole_word", "substring"],
    case_sensitive: bool,
) -> re.Pattern[str]:
    if match_mode == "whole_word":
        escaped = r"\s+".join(re.escape(part) for part in query.split())
        prefix = r"(?<!\w)" if query[0].isalnum() or query[0] == "_" else ""
        suffix = r"(?!\w)" if query[-1].isalnum() or query[-1] == "_" else ""
        expression = f"{prefix}{escaped}{suffix}"
    elif match_mode == "substring":
        expression = re.escape(query)
    else:
        raise ValueError(f"unsupported match_mode: {match_mode}")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(expression, flags=flags)


def _match_context(text: str, start: int, end: int) -> str:
    radius = MAX_OCCURRENCE_CONTEXT_LENGTH // 2
    excerpt_start = max(0, start - radius)
    excerpt_end = min(len(text), end + radius)
    excerpt = " ".join(text[excerpt_start:excerpt_end].split())
    if excerpt_start > 0:
        excerpt = "..." + excerpt
    if excerpt_end < len(text):
        excerpt += "..."
    return excerpt
