"""On-demand PDF page and figure analysis through the configured VLM."""

import math
import os
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from PIL import Image

from file_agent.agent_tools.common import (
    _block_page_number,
    _evidence_command,
    _find_document,
    _source_metadata,
)
from file_agent.agent_tools.limits import (
    MAX_VISUAL_CONTEXT_LENGTH,
    MAX_VISUAL_DESCRIPTION_LENGTH,
    MAX_VISUAL_PIXELS,
    MAX_VISUAL_QUESTION_LENGTH,
    VISUAL_CROP_PADDING,
)
from file_agent.agent_tools.prompts import VISUAL_ANALYSIS_PROMPT
from file_agent.document import Block, BlockType, Document
from file_agent.utils.image_extractor import extract_image_from_pdf_bytes


@tool
def analyze_document_visual(
    source_file: str,
    question: str,
    runtime: ToolRuntime[Any, dict],
    visual_id: str | None = None,
    page_number: int | None = None,
) -> Command:
    """Analyze a PDF figure or page with the configured vision-language model.

    Provide exactly one target. Prefer visual_id from get_document_outline for a
    precise crop. Use page_number as a fallback when the visual was not detected.

    Args:
        source_file: Exact PDF file name returned by list_documents.
        question: Specific question to answer about the visible content.
        visual_id: Exact visual_id returned by get_document_outline.
        page_number: One-indexed PDF page to analyze as a whole-page fallback.
    """
    document = _find_document(runtime.context.documents, source_file)
    if document.file_type.lower().lstrip(".") != "pdf":
        raise ValueError("visual analysis currently supports PDF documents only")

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")
    if len(normalized_question) > MAX_VISUAL_QUESTION_LENGTH:
        raise ValueError(f"question must not exceed {MAX_VISUAL_QUESTION_LENGTH} characters")

    normalized_visual_id = visual_id.strip() if visual_id is not None else None
    has_visual_id = bool(normalized_visual_id)
    has_page_number = page_number is not None
    if has_visual_id == has_page_number:
        raise ValueError("provide exactly one of visual_id or page_number")

    vlm_client = runtime.context.vlm_client
    if vlm_client is None:
        raise ValueError("visual analysis is unavailable: no VLM backend is configured")
    asset_store = runtime.context.asset_store
    if asset_store is None:
        raise ValueError("visual analysis is unavailable: original document asset is missing")

    selected_block: Block | None = None
    bbox: tuple[float, float, float, float] | None = None
    if has_visual_id:
        selected_block = next(
            (
                block
                for block in document.blocks
                if block.id == normalized_visual_id
                and block.block_type in (BlockType.FIGURE, BlockType.IMAGE)
            ),
            None,
        )
        if selected_block is None:
            raise ValueError(
                f"visual is not indexed in {document.file_name}: {normalized_visual_id}"
            )
        block_page_number = _block_page_number(selected_block)
        if block_page_number is None or selected_block.bbox is None:
            raise ValueError("the selected visual has no renderable PDF coordinates")
        resolved_page_number = block_page_number
        bbox = selected_block.bbox
        target = f"visual_id={selected_block.id}"
    else:
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number <= 0:
            raise ValueError("page_number must be a positive integer")
        resolved_page_number = page_number
        target = "whole PDF page"

    max_visual_pixels = _resolve_max_visual_pixels()
    image = extract_image_from_pdf_bytes(
        asset_store.get_bytes(document.file_name),
        page_number=resolved_page_number,
        bbox=bbox,
        padding=VISUAL_CROP_PADDING if bbox is not None else 0.0,
        max_pixels=max_visual_pixels,
    )
    if image is None:
        raise ValueError("the selected PDF visual could not be rendered")
    image = _limit_image_pixels(image, max_visual_pixels)

    relevant_blocks = (
        [selected_block]
        if selected_block is not None
        else [
            block for block in document.blocks if _block_page_number(block) == resolved_page_number
        ]
    )
    existing_description = selected_block.vlm_description if selected_block is not None else None
    prompt = VISUAL_ANALYSIS_PROMPT.format(
        source_file=document.file_name,
        page_number=resolved_page_number,
        target=target,
        existing_description=existing_description or "not available",
        nearby_text=_visual_nearby_text(
            document,
            resolved_page_number,
            selected_block,
        ),
        question=normalized_question,
    )
    analysis = vlm_client.describe_image(image, prompt).strip()
    if not analysis:
        raise ValueError("VLM returned an empty visual analysis")

    metadata = _source_metadata(document, relevant_blocks)
    metadata.update(
        {
            "page_number": resolved_page_number,
            "visual_id": selected_block.id if selected_block is not None else None,
            "bbox": bbox,
            "evidence_type": "visual_analysis",
        }
    )
    payload = {
        "source_file": document.file_name,
        "page_number": resolved_page_number,
        "visual_id": metadata["visual_id"],
        "bbox": bbox,
        "analysis": analysis,
        "metadata": metadata,
    }
    evidence_target = selected_block.id if selected_block is not None else f"page-{page_number}"
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"visual:{document.file_name}:{evidence_target}",
        text=analysis,
        metadata=metadata,
    )


def _document_visuals(document: Document) -> list[dict[str, Any]]:
    visuals = []
    current_section: str | None = None
    for block in document.blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            current_section = block.text.strip()
        if block.block_type not in (BlockType.FIGURE, BlockType.IMAGE):
            continue

        description = block.vlm_description or block.text.strip() or None
        if description and len(description) > MAX_VISUAL_DESCRIPTION_LENGTH:
            description = description[:MAX_VISUAL_DESCRIPTION_LENGTH].rstrip() + "..."
        visuals.append(
            {
                "visual_id": block.id,
                "block_type": block.block_type.value,
                "section": current_section,
                "page_number": _block_page_number(block),
                "bbox": block.bbox,
                "description": description,
                "renderable": _block_page_number(block) is not None and block.bbox is not None,
            }
        )
    return visuals


def _visual_nearby_text(
    document: Document,
    page_number: int,
    selected_block: Block | None,
) -> str:
    page_blocks = [
        block
        for block in document.blocks
        if _block_page_number(block) == page_number and block is not selected_block
    ]
    parts = []
    for block in page_blocks:
        text = block.text.strip()
        if not text or block.block_type in (BlockType.FIGURE, BlockType.IMAGE):
            continue
        parts.append(text)
    nearby_text = "\n\n".join(parts).strip()
    if not nearby_text:
        return "not available"
    if len(nearby_text) > MAX_VISUAL_CONTEXT_LENGTH:
        return nearby_text[:MAX_VISUAL_CONTEXT_LENGTH].rstrip() + "..."
    return nearby_text


def _limit_image_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    pixels = image.width * image.height
    if pixels <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / pixels)
    size = (
        max(1, math.floor(image.width * scale)),
        max(1, math.floor(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _resolve_max_visual_pixels() -> int:
    raw_value = os.getenv("VLM_MAX_VISUAL_PIXELS", str(MAX_VISUAL_PIXELS))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("VLM_MAX_VISUAL_PIXELS must be an integer") from exc
    if value < 1:
        raise ValueError("VLM_MAX_VISUAL_PIXELS must be greater than zero")
    return value
