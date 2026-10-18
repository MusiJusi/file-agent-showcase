"""Deterministic table lookup, aggregation, comparison, and arithmetic."""

import hashlib
import json
import re
from decimal import Decimal, DecimalException, InvalidOperation
from typing import Any, Literal

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

from file_agent.agent_tools.common import (
    _block_page_number,
    _evidence_command,
    _find_document,
    _paginate_rows,
    _source_metadata,
    _tool_command,
)
from file_agent.agent_tools.limits import (
    MAX_CALCULATION_VALUES,
    MAX_TABLE_ANALYSIS_RESULTS,
    MAX_TABLE_LOOKUP_RESULTS,
)
from file_agent.chunking import Chunk
from file_agent.document import Block, BlockType, Document
from file_agent.retrieval import SearchResult


@tool
def read_table(
    source_file: str,
    table_id: str,
    runtime: ToolRuntime[Any, dict],
    offset: int = 0,
    limit: int = 25,
) -> Command:
    """Read rows from a table or XLSX sheet identified by get_document_outline.

    The payload distinguishes ``data_rows`` (header excluded) from
    ``total_rows`` (all readable lines used by pagination).

    Args:
        source_file: Exact file name returned by list_documents.
        table_id: Exact table_id returned by get_document_outline.
        offset: Zero-based row offset returned as next_offset.
        limit: Maximum number of rows to return, from 1 to 50.
    """
    document = _find_document(runtime.context.documents, source_file)
    normalized_id = table_id.strip()
    block = next(
        (
            item
            for item in document.blocks
            if item.id == normalized_id and _is_table_block(document, item)
        ),
        None,
    )
    if block is None:
        raise ValueError(f"table is not indexed in {document.file_name}: {table_id}")

    rows = [line for line in block.text.splitlines() if line.strip()]
    row_page = _paginate_rows(rows, offset, limit)
    metadata = _source_metadata(document, [block])
    metadata.update(
        {
            "table_id": normalized_id,
            "row_offset": offset,
            "table_format": _table_format(document, block),
        }
    )
    payload = {
        "source_file": document.file_name,
        "table_id": normalized_id,
        "format": metadata["table_format"],
        **row_page,
        "data_rows": _table_data_row_count(document, block),
        "metadata": metadata,
    }
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"table:{document.file_name}:{normalized_id}:{offset}",
        text="\n".join(row_page["rows"]),
        metadata=metadata,
    )


@tool
def analyze_table(
    source_file: str,
    table_id: str,
    operation: Literal["count_distinct", "sum", "average", "min", "max", "group_sum"],
    value_column: str,
    runtime: ToolRuntime[Any, dict],
    group_by: str | None = None,
    multiply_by: str | None = None,
    top_n: int = 5,
) -> Command:
    """Calculate an exact result over every row of a table or XLSX sheet.

    Prefer this tool over read_table when the question asks for a total, unique
    count, sum/average, minimum/maximum row, or grouped total. ``multiply_by`` calculates a
    row value as ``value_column * multiply_by`` (for example stock * price).

    Args:
        source_file: Exact file name returned by list_documents.
        table_id: Exact table_id returned by get_document_outline.
        operation: count_distinct, sum, average, min, max, or group_sum.
        value_column: Column to count or calculate.
        group_by: Required grouping column for group_sum.
        multiply_by: Optional second numeric column multiplied into each value.
        top_n: Number of ranked rows/groups to return, from 1 to 20.
    """
    document = _find_document(runtime.context.documents, source_file)
    normalized_id = table_id.strip()
    block = next(
        (
            item
            for item in document.blocks
            if item.id == normalized_id and _is_table_block(document, item)
        ),
        None,
    )
    if block is None:
        raise ValueError(f"table is not indexed in {document.file_name}: {table_id}")

    headers, rows = _parse_table_rows(document, block)
    value_key = _resolve_table_column(headers, value_column)
    group_key = _resolve_table_column(headers, group_by) if group_by is not None else None
    multiplier_key = (
        _resolve_table_column(headers, multiply_by) if multiply_by is not None else None
    )
    bounded_top_n = _bounded_table_analysis_results(top_n)

    result, rows_used, rows_skipped = _calculate_table_result(
        rows=rows,
        operation=operation,
        value_key=value_key,
        group_key=group_key,
        multiplier_key=multiplier_key,
        top_n=bounded_top_n,
    )
    metadata = _source_metadata(document, [block])
    metadata.update(
        {
            "table_id": normalized_id,
            "table_format": _table_format(document, block),
            "table_operation": operation,
        }
    )
    payload = {
        "source_file": document.file_name,
        "table_id": normalized_id,
        "operation": operation,
        "value_column": value_key,
        "multiply_by": multiplier_key,
        "group_by": group_key,
        "total_data_rows": len(rows),
        "rows_used": rows_used,
        "rows_skipped": rows_skipped,
        "result": result,
        "metadata": metadata,
    }
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"table-analysis:{document.file_name}:{normalized_id}:{operation}",
        text=evidence_text,
        metadata=metadata,
    )


@tool
def lookup_table_rows(
    source_file: str,
    table_id: str,
    match_column: str,
    match_value: str,
    runtime: ToolRuntime[Any, dict],
    match_mode: Literal["exact", "contains"] = "exact",
    limit: int = 10,
) -> Command:
    """Return complete rows matching a value in one exact table column.

    Matching is whitespace-normalized and case-insensitive. Use this for row and
    column lookup questions so a value cannot drift to an adjacent row.

    Args:
        source_file: Exact file name returned by list_documents.
        table_id: Exact table_id returned by get_document_outline.
        match_column: Exact column containing the row label or lookup key.
        match_value: Value to locate in that column.
        match_mode: exact or contains.
        limit: Maximum matching rows returned, from 1 to 20.
    """
    document = _find_document(runtime.context.documents, source_file)
    block = _find_table_block(document, table_id)
    headers, rows = _parse_table_rows(document, block)
    match_key = _resolve_table_column(headers, match_column)
    normalized_value = " ".join(match_value.split()).casefold()
    if not normalized_value:
        raise ValueError("match_value must not be empty")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    bounded_limit = max(1, min(limit, MAX_TABLE_LOOKUP_RESULTS))

    def matches(row: dict[str, str]) -> bool:
        candidate = " ".join(row[match_key].split()).casefold()
        if match_mode == "exact":
            return candidate == normalized_value
        return normalized_value in candidate

    all_matches = [row for row in rows if matches(row)]
    metadata = _source_metadata(document, [block])
    metadata.update(
        {
            "table_id": block.id,
            "table_format": _table_format(document, block),
            "table_operation": "lookup_rows",
        }
    )
    payload = {
        "source_file": document.file_name,
        "table_id": block.id,
        "headers": headers,
        "match_column": match_key,
        "match_value": match_value,
        "match_mode": match_mode,
        "total_data_rows": len(rows),
        "matching_rows_count": len(all_matches),
        "rows": all_matches[:bounded_limit],
        "rows_truncated": len(all_matches) > bounded_limit,
        "metadata": metadata,
    }
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    digest = hashlib.sha256(normalized_value.encode()).hexdigest()[:12]
    return _evidence_command(
        payload,
        runtime,
        evidence_id=f"table-lookup:{document.file_name}:{block.id}:{digest}",
        text=evidence_text,
        metadata=metadata,
    )


@tool
def compare_table_columns(
    left_source_file: str,
    left_table_id: str,
    left_column: str,
    right_source_file: str,
    right_table_id: str,
    right_column: str,
    operation: Literal["intersection", "left_only", "right_only", "union"],
    runtime: ToolRuntime[Any, dict],
    top_n: int = 10,
) -> Command:
    """Compare distinct values from columns in two complete tables or XLSX sheets.

    Use this for exact cross-file or cross-sheet overlap and difference questions.
    Values are trimmed, whitespace-normalized, and compared case-insensitively.
    The count covers the complete tables; ``top_n`` only limits returned examples.

    Args:
        left_source_file: Exact first file name returned by list_documents.
        left_table_id: Exact first table_id returned by get_document_outline.
        left_column: Column from the first table to compare.
        right_source_file: Exact second file name returned by list_documents.
        right_table_id: Exact second table_id returned by get_document_outline.
        right_column: Column from the second table to compare.
        operation: intersection, left_only, right_only, or union.
        top_n: Number of example values to return, from 1 to 20.
    """
    left_document = _find_document(runtime.context.documents, left_source_file)
    right_document = _find_document(runtime.context.documents, right_source_file)
    left_block = _find_table_block(left_document, left_table_id)
    right_block = _find_table_block(right_document, right_table_id)

    left_headers, left_rows = _parse_table_rows(left_document, left_block)
    right_headers, right_rows = _parse_table_rows(right_document, right_block)
    left_key = _resolve_table_column(left_headers, left_column)
    right_key = _resolve_table_column(right_headers, right_column)
    bounded_top_n = _bounded_table_analysis_results(top_n)

    left_values = _distinct_table_values(left_rows, left_key)
    right_values = _distinct_table_values(right_rows, right_key)
    result_values = _compare_distinct_values(left_values, right_values, operation)
    result_labels = [result_values[key] for key in sorted(result_values)]
    examples = result_labels[:bounded_top_n]

    left_summary = {
        "source_file": left_document.file_name,
        "table_id": left_block.id,
        "column": left_key,
        "total_data_rows": len(left_rows),
        "distinct_non_empty_values": len(left_values),
    }
    right_summary = {
        "source_file": right_document.file_name,
        "table_id": right_block.id,
        "column": right_key,
        "total_data_rows": len(right_rows),
        "distinct_non_empty_values": len(right_values),
    }
    payload = {
        "operation": operation,
        "normalization": "trim_whitespace_casefold",
        "left": left_summary,
        "right": right_summary,
        "result": {
            "count": len(result_labels),
            "examples": examples,
            "examples_truncated": len(result_labels) > len(examples),
        },
    }
    evidence_text = json.dumps(payload, ensure_ascii=False, default=str)
    sources = _table_comparison_sources(
        left_document=left_document,
        left_block=left_block,
        right_document=right_document,
        right_block=right_block,
        operation=operation,
        evidence_text=evidence_text,
    )
    return _tool_command(payload, runtime, sources=sources)


@tool
def calculate(
    operation: Literal[
        "add",
        "subtract",
        "multiply",
        "divide",
        "average",
        "percentage_of",
        "percent_change",
    ],
    values: list[float],
) -> str:
    """Perform deterministic decimal arithmetic over document-derived values.

    This tool does not read documents and is not evidence. Read the source values
    with a document tool first. ``subtract`` and ``divide`` calculate first versus
    second; ``percentage_of`` calculates first / second * 100; ``percent_change``
    calculates the change from the first (old) value to the second (new) value.

    Args:
        operation: Arithmetic operation to perform.
        values: Numeric input values in operation order. Binary operations require
            exactly two values; add, multiply, and average accept one or more.
    """
    decimal_values = [Decimal(str(value)) for value in values]
    result, unit = _calculate_values(operation, decimal_values)
    return json.dumps(
        {
            "operation": operation,
            "values": [_format_decimal(value) for value in decimal_values],
            "result": _format_decimal(result),
            "unit": unit,
        },
        ensure_ascii=False,
    )


def _document_tables(document: Document) -> list[dict[str, Any]]:
    tables = []
    current_section: str | None = None
    for block in document.blocks:
        if block.block_type == BlockType.HEADING and block.text.strip():
            current_section = block.text.strip()
        if not _is_table_block(document, block):
            continue
        readable_rows = len([line for line in block.text.splitlines() if line.strip()])
        tables.append(
            {
                "table_id": block.id,
                "section": current_section,
                "page_number": _block_page_number(block),
                "slide_number": block.metadata.get("slide_number"),
                "sheet_name": block.metadata.get("sheet_name"),
                "rows": _table_data_row_count(document, block),
                "readable_rows_including_header": readable_rows,
                "format": _table_format(document, block),
            }
        )
    return tables


def _is_table_block(document: Document, block: Block) -> bool:
    file_type = document.file_type.lower().lstrip(".")
    return (
        block.block_type == BlockType.TABLE
        or block.type == BlockType.TABLE.value
        or (file_type == "xlsx" and block.type == "xlsx_sheet")
    )


def _find_table_block(document: Document, table_id: str) -> Block:
    normalized_id = table_id.strip()
    if not normalized_id:
        raise ValueError("table_id must not be empty")
    block = next(
        (
            item
            for item in document.blocks
            if item.id == normalized_id and _is_table_block(document, item)
        ),
        None,
    )
    if block is None:
        raise ValueError(f"table is not indexed in {document.file_name}: {table_id}")
    return block


def _table_format(document: Document, block: Block) -> str:
    lines = [line for line in block.text.splitlines() if line.strip()]
    if len(lines) >= 2 and "|" in lines[0] and "|" in lines[1]:
        separator = [cell.strip() for cell in lines[1].strip().strip("|").split("|")]
        if _is_markdown_separator_row(separator):
            return "markdown"
    if "\t" in block.text:
        return "tsv"
    if "|" in block.text:
        return "markdown"
    if document.file_type.lower().lstrip(".") == "xlsx":
        return "tsv"
    return "text"


def _parse_table_rows(
    document: Document,
    block: Block,
) -> tuple[list[str], list[dict[str, str]]]:
    lines = [line for line in block.text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("the selected table must contain a header and at least one data row")

    table_format = _table_format(document, block)
    if table_format == "tsv":
        parsed = [[cell.strip() for cell in line.split("\t")] for line in lines]
    elif table_format == "markdown":
        parsed = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
        parsed = [row for row in parsed if not _is_markdown_separator_row(row)]
    else:
        raise ValueError("table analysis currently supports TSV/XLSX and Markdown tables")

    headers = parsed[0]
    if not headers or any(not header for header in headers):
        raise ValueError("the selected table has an invalid or empty header")
    if len({header.casefold() for header in headers}) != len(headers):
        raise ValueError("the selected table contains duplicate column names")

    rows = []
    for values in parsed[1:]:
        padded = [*values[: len(headers)], *([""] * max(0, len(headers) - len(values)))]
        rows.append(dict(zip(headers, padded, strict=True)))
    if not rows:
        raise ValueError("the selected table contains no data rows")
    return headers, rows


def _table_data_row_count(document: Document, block: Block) -> int:
    try:
        _, rows = _parse_table_rows(document, block)
    except ValueError:
        return len([line for line in block.text.splitlines() if line.strip()])
    return len(rows)


def _is_markdown_separator_row(values: list[str]) -> bool:
    return bool(values) and all(re.fullmatch(r":?-{3,}:?", value) for value in values)


def _resolve_table_column(headers: list[str], requested: str | None) -> str:
    if requested is None or not requested.strip():
        raise ValueError("column name must not be empty")
    normalized = requested.strip().casefold()
    match = next((header for header in headers if header.casefold() == normalized), None)
    if match is None:
        raise ValueError(f"column is not available: {requested}; available columns: {headers}")
    return match


def _bounded_table_analysis_results(top_n: int) -> int:
    if not isinstance(top_n, int) or isinstance(top_n, bool):
        raise ValueError("top_n must be an integer")
    return max(1, min(top_n, MAX_TABLE_ANALYSIS_RESULTS))


def _distinct_table_values(
    rows: list[dict[str, str]],
    column: str,
) -> dict[str, str]:
    distinct: dict[str, str] = {}
    for row in rows:
        display_value = " ".join(row[column].split())
        if display_value:
            distinct.setdefault(display_value.casefold(), display_value)
    return distinct


def _compare_distinct_values(
    left_values: dict[str, str],
    right_values: dict[str, str],
    operation: str,
) -> dict[str, str]:
    left_keys = set(left_values)
    right_keys = set(right_values)
    if operation == "intersection":
        result_keys = left_keys & right_keys
    elif operation == "left_only":
        result_keys = left_keys - right_keys
    elif operation == "right_only":
        result_keys = right_keys - left_keys
    elif operation == "union":
        result_keys = left_keys | right_keys
    else:
        raise ValueError(f"unsupported table comparison operation: {operation}")

    return {
        key: left_values[key] if key in left_values else right_values[key] for key in result_keys
    }


def _table_comparison_sources(
    left_document: Document,
    left_block: Block,
    right_document: Document,
    right_block: Block,
    operation: str,
    evidence_text: str,
) -> list[SearchResult]:
    comparison_id = (
        f"{left_document.file_name}:{left_block.id}:"
        f"{right_document.file_name}:{right_block.id}:{operation}"
    )
    sources = []
    for role, document, block, other_document in (
        ("left", left_document, left_block, right_document),
        ("right", right_document, right_block, left_document),
    ):
        metadata = _source_metadata(document, [block])
        metadata.update(
            {
                "table_id": block.id,
                "table_format": _table_format(document, block),
                "table_operation": f"compare_{operation}",
                "comparison_role": role,
                "compared_source_file": other_document.file_name,
            }
        )
        sources.append(
            SearchResult(
                chunk=Chunk(
                    id=f"table-comparison:{comparison_id}:{role}",
                    text=evidence_text,
                    metadata=metadata,
                ),
                score=1.0,
            )
        )
    return sources


def _calculate_values(
    operation: str,
    values: list[Decimal],
) -> tuple[Decimal, str | None]:
    if not values:
        raise ValueError("values must contain at least one number")
    if len(values) > MAX_CALCULATION_VALUES:
        raise ValueError(f"values must not contain more than {MAX_CALCULATION_VALUES} numbers")
    if any(not value.is_finite() for value in values):
        raise ValueError("values must contain only finite numbers")

    binary_operations = {"subtract", "divide", "percentage_of", "percent_change"}
    if operation in binary_operations and len(values) != 2:
        raise ValueError(f"{operation} requires exactly two values")

    try:
        if operation == "add":
            result = sum(values, Decimal(0))
        elif operation == "subtract":
            result = values[0] - values[1]
        elif operation == "multiply":
            result = Decimal(1)
            for value in values:
                result *= value
        elif operation == "divide":
            if values[1] == 0:
                raise ValueError("cannot divide by zero")
            result = values[0] / values[1]
        elif operation == "average":
            result = sum(values, Decimal(0)) / Decimal(len(values))
        elif operation == "percentage_of":
            if values[1] == 0:
                raise ValueError("the whole value must not be zero")
            result = values[0] / values[1] * Decimal(100)
        elif operation == "percent_change":
            if values[0] == 0:
                raise ValueError("the old value must not be zero")
            result = (values[1] - values[0]) / values[0] * Decimal(100)
        else:
            raise ValueError(f"unsupported calculation operation: {operation}")
    except DecimalException as exc:
        raise ValueError("calculation could not be represented as a finite decimal") from exc

    if not result.is_finite():
        raise ValueError("calculation result must be finite")
    unit = "percent" if operation in {"percentage_of", "percent_change"} else None
    return result, unit


def _calculate_table_result(
    rows: list[dict[str, str]],
    operation: str,
    value_key: str,
    group_key: str | None,
    multiplier_key: str | None,
    top_n: int,
) -> tuple[dict[str, Any], int, int]:
    if operation == "count_distinct":
        if group_key is not None or multiplier_key is not None:
            raise ValueError("count_distinct does not accept group_by or multiply_by")
        distinct: dict[str, str] = {}
        for row in rows:
            value = row[value_key].strip()
            if value:
                distinct.setdefault(value.casefold(), value)
        values = sorted(distinct.values(), key=str.casefold)
        return {"count": len(values), "values": values}, len(rows), 0

    if operation == "group_sum" and group_key is None:
        raise ValueError("group_by is required for group_sum")
    if operation != "group_sum" and group_key is not None:
        raise ValueError("group_by is only supported for group_sum")

    calculated: list[tuple[Decimal, dict[str, str]]] = []
    skipped = 0
    for row in rows:
        try:
            value = _decimal_value(row[value_key])
            if multiplier_key is not None:
                value *= _decimal_value(row[multiplier_key])
        except ValueError:
            skipped += 1
            continue
        calculated.append((value, row))
    if not calculated:
        raise ValueError("the selected numeric columns contain no usable values")

    if operation == "sum":
        total = sum((value for value, _ in calculated), Decimal(0))
        return {"value": _format_decimal(total)}, len(calculated), skipped

    if operation == "average":
        average = sum((value for value, _ in calculated), Decimal(0)) / Decimal(len(calculated))
        return {"value": _format_decimal(average)}, len(calculated), skipped

    if operation in ("min", "max"):
        reverse = operation == "max"
        ranked = sorted(calculated, key=lambda item: item[0], reverse=reverse)[:top_n]
        return (
            {"rows": [{"value": _format_decimal(value), "row": row} for value, row in ranked]},
            len(calculated),
            skipped,
        )

    if operation == "group_sum":
        grouped: dict[str, tuple[str, Decimal]] = {}
        for value, row in calculated:
            group = row[group_key].strip()  # type: ignore[index]
            if not group:
                skipped += 1
                continue
            normalized_group = group.casefold()
            label, total = grouped.get(normalized_group, (group, Decimal(0)))
            grouped[normalized_group] = (label, total + value)
        ranked_groups = sorted(grouped.values(), key=lambda item: item[1], reverse=True)[:top_n]
        return (
            {
                "groups": [
                    {"group": label, "value": _format_decimal(value)}
                    for label, value in ranked_groups
                ]
            },
            len(calculated),
            skipped,
        )

    raise ValueError(f"unsupported table operation: {operation}")


def _decimal_value(raw: str) -> Decimal:
    normalized = raw.strip().replace("\u00a0", "").replace(" ", "")
    normalized = normalized.removeprefix("$").removeprefix("€").removeprefix("£")
    normalized = normalized.removesuffix("₽").removesuffix("%").strip()
    if not normalized:
        raise ValueError("numeric value is empty")
    if "," in normalized and "." in normalized:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif "," in normalized:
        normalized = normalized.replace(",", ".")
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise ValueError(f"not a numeric value: {raw}") from exc
    if not value.is_finite():
        raise ValueError(f"not a finite numeric value: {raw}")
    return value


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"
