"""Test table tools."""

import json

import pytest
from langchain_core.messages import AIMessage

from file_agent.agent_tools import DOCUMENT_TOOLS, calculate
from file_agent.document import Block, BlockType, Document
from tests.agent._support import (
    FakeRetriever,
    FakeToolCallingLLM,
    _invoke_single_read_tool,
    invoke_tool_agent,
    make_search_result,
    tool_messages_seen_by_model,
)


def test_table_comparison_tool_schema_is_explicit():
    comparison_tool = next(tool for tool in DOCUMENT_TOOLS if tool.name == "compare_table_columns")

    assert set(comparison_tool.tool_call_schema.model_json_schema()["properties"]) == {
        "left_source_file",
        "left_table_id",
        "left_column",
        "right_source_file",
        "right_table_id",
        "right_column",
        "operation",
        "top_n",
    }


def test_xlsx_question_routes_to_outline_even_without_table_wording():
    document = Document(
        file_name="inventory.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text="product\tprice\nBeta\t20",
                type="xlsx_sheet",
                metadata={"sheet_name": "Inventory"},
                block_type=BlockType.TABLE,
            )
        ],
    )
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_document_outline",
                        "args": {"source_file": "inventory.xlsx"},
                        "id": "outline-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_table_rows",
                        "args": {
                            "source_file": "inventory.xlsx",
                            "table_id": "sheet-1",
                            "match_column": "product",
                            "match_value": "Beta",
                        },
                        "id": "lookup-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Beta's price is 20."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever(),
        documents=[document],
        require_evidence_tool=True,
        question="What is Beta's price?",
    )

    assert llm_client.calls[0]["tool_names"] == ["get_document_outline"]
    assert state["response"].answer == "Beta's price is 20."


def test_calculate_does_not_count_as_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {"operation": "add", "values": [2, 2]},
                        "id": "calculate-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="4"),
        ]
    )

    with pytest.raises(ValueError, match="without successfully using"):
        invoke_tool_agent(
            llm_client,
            FakeRetriever(),
            require_evidence_tool=True,
        )

    assert [call["tool_choice"] for call in llm_client.calls] == ["required", "required"]


def test_tool_agent_can_read_table_rows_with_pagination():
    document = Document(
        file_name="budget.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text="Name\tValue\nA\t1\nB\t2\nC\t3",
                type="xlsx_sheet",
                metadata={"sheet_name": "Budget"},
                block_type=BlockType.TABLE,
            )
        ],
    )

    state, llm_client = _invoke_single_read_tool(
        "read_table",
        {
            "source_file": "budget.xlsx",
            "table_id": "sheet-1",
            "offset": 1,
            "limit": 2,
        },
        [document],
    )

    tool_message = tool_messages_seen_by_model(llm_client)[0]
    payload = json.loads(str(tool_message.content))
    assert payload["format"] == "tsv"
    assert payload["rows"] == ["A\t1", "B\t2"]
    assert payload["next_offset"] == 3
    assert payload["total_rows"] == 4
    assert payload["data_rows"] == 3
    assert state["response"].sources[0].chunk.metadata["table_id"] == "sheet-1"


@pytest.mark.parametrize(
    ("arguments", "expected_result"),
    [
        (
            {
                "operation": "count_distinct",
                "value_column": "category",
            },
            {"count": 2, "values": ["X", "Y"]},
        ),
        (
            {
                "operation": "sum",
                "value_column": "stock",
                "multiply_by": "price",
            },
            {"value": "180"},
        ),
        (
            {
                "operation": "average",
                "value_column": "stock",
            },
            {"value": "2"},
        ),
        (
            {
                "operation": "min",
                "value_column": "stock",
                "top_n": 1,
            },
            {
                "rows": [
                    {
                        "value": "1",
                        "row": {
                            "product": "C",
                            "stock": "1",
                            "price": "100",
                            "category": "Y",
                            "amount": "12",
                            "region": "North",
                        },
                    }
                ]
            },
        ),
        (
            {
                "operation": "group_sum",
                "value_column": "stock",
                "multiply_by": "price",
                "group_by": "category",
                "top_n": 2,
            },
            {
                "groups": [
                    {"group": "Y", "value": "100"},
                    {"group": "X", "value": "80"},
                ]
            },
        ),
    ],
)
def test_tool_agent_can_analyze_all_table_rows(arguments, expected_result):
    document = Document(
        file_name="inventory.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text=(
                    "product\tstock\tprice\tcategory\tamount\tregion\n"
                    "A\t2\t10\tX\t5\tNorth\n"
                    "B\t3\t20\tX\t8\tSouth\n"
                    "C\t1\t100\tY\t12\tNorth"
                ),
                type="xlsx_sheet",
                metadata={"sheet_name": "Inventory"},
                block_type=BlockType.TABLE,
            )
        ],
    )
    state, llm_client = _invoke_single_read_tool(
        "analyze_table",
        {
            "source_file": "inventory.xlsx",
            "table_id": "sheet-1",
            **arguments,
        },
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["total_data_rows"] == 3
    assert payload["rows_skipped"] == 0
    assert payload["result"] == expected_result
    assert state["response"].sources[0].chunk.metadata["table_operation"] == arguments["operation"]


def test_tool_agent_can_lookup_an_exact_table_row():
    document = Document(
        file_name="inventory.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-1",
                text="product\tstock\tprice\nAlpha\t2\t10\nBeta\t3\t20",
                type="xlsx_sheet",
                metadata={"sheet_name": "Inventory"},
                block_type=BlockType.TABLE,
            )
        ],
    )
    state, llm_client = _invoke_single_read_tool(
        "lookup_table_rows",
        {
            "source_file": "inventory.xlsx",
            "table_id": "sheet-1",
            "match_column": "product",
            "match_value": " beta ",
        },
        [document],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["matching_rows_count"] == 1
    assert payload["headers"] == ["product", "stock", "price"]
    assert payload["rows"] == [{"product": "Beta", "stock": "3", "price": "20"}]
    assert state["response"].sources[0].chunk.metadata["table_operation"] == "lookup_rows"


@pytest.mark.parametrize(
    ("operation", "expected_count", "expected_examples"),
    [
        (
            "intersection",
            3,
            ["Luxurious Steel Chips", "Practical Steel Bike", "Rustic Gold Shoes"],
        ),
        ("left_only", 1, ["Only Inventory"]),
        ("right_only", 1, ["Only Sales"]),
        (
            "union",
            5,
            [
                "Luxurious Steel Chips",
                "Only Inventory",
                "Only Sales",
                "Practical Steel Bike",
                "Rustic Gold Shoes",
            ],
        ),
    ],
)
def test_tool_agent_compares_complete_columns_across_tables(
    operation,
    expected_count,
    expected_examples,
):
    inventory = Document(
        file_name="inventory.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-inventory",
                text=(
                    "| product | stock |\n"
                    "| --- | --- |\n"
                    "| Practical Steel Bike | 2 |\n"
                    "| Rustic  Gold Shoes | 3 |\n"
                    "| Luxurious Steel Chips | 1 |\n"
                    "| Only Inventory | 8 |\n"
                    "| practical steel bike | 5 |"
                ),
                type="xlsx_sheet",
                metadata={
                    "sheet_name": "Inventory",
                    "dataset_doc_id": "q0122/inventory.xlsx",
                },
                block_type=BlockType.TABLE,
            )
        ],
    )
    sales = Document(
        file_name="sales.xlsx",
        file_type="xlsx",
        blocks=[
            Block(
                id="sheet-sales",
                text=(
                    "| item | amount |\n"
                    "| --- | --- |\n"
                    "| practical steel bike | 10 |\n"
                    "| Rustic Gold Shoes | 20 |\n"
                    "| LUXURIOUS STEEL CHIPS | 30 |\n"
                    "| Only Sales | 40 |"
                ),
                type="xlsx_sheet",
                metadata={
                    "sheet_name": "Sales",
                    "dataset_doc_id": "q0122/sales.xlsx",
                },
                block_type=BlockType.TABLE,
            )
        ],
    )

    state, llm_client = _invoke_single_read_tool(
        "compare_table_columns",
        {
            "left_source_file": "inventory.xlsx",
            "left_table_id": "sheet-inventory",
            "left_column": "product",
            "right_source_file": "sales.xlsx",
            "right_table_id": "sheet-sales",
            "right_column": "item",
            "operation": operation,
        },
        [inventory, sales],
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[0].content))
    assert payload["left"]["total_data_rows"] == 5
    assert payload["left"]["distinct_non_empty_values"] == 4
    assert payload["right"]["total_data_rows"] == 4
    assert payload["right"]["distinct_non_empty_values"] == 4
    assert payload["left"]["column"] == "product"
    assert payload["right"]["column"] == "item"
    assert payload["result"] == {
        "count": expected_count,
        "examples": expected_examples,
        "examples_truncated": False,
    }
    sources = state["response"].sources
    assert len(sources) == 2
    assert {source.chunk.metadata["dataset_doc_id"] for source in sources} == {
        "q0122/inventory.xlsx",
        "q0122/sales.xlsx",
    }
    assert {source.chunk.metadata["table_operation"] for source in sources} == {
        f"compare_{operation}"
    }


@pytest.mark.parametrize(
    ("operation", "values", "expected_result", "expected_unit"),
    [
        ("add", ["100.1", "20.2"], "120.3", None),
        ("subtract", ["150", "120"], "30", None),
        ("multiply", ["12.5", "4"], "50", None),
        ("divide", ["150", "120"], "1.25", None),
        ("average", ["10", "20", "30"], "20", None),
        ("percentage_of", ["30", "120"], "25", "percent"),
        ("percent_change", ["120", "150"], "25", "percent"),
    ],
)
def test_calculate_performs_decimal_arithmetic(
    operation,
    values,
    expected_result,
    expected_unit,
):
    payload = json.loads(calculate.invoke({"operation": operation, "values": values}))

    assert payload == {
        "operation": operation,
        "values": values,
        "result": expected_result,
        "unit": expected_unit,
    }


def test_calculate_uses_vllm_compatible_numeric_schema():
    schema = calculate.args_schema.model_json_schema()

    assert schema["properties"]["values"]["items"] == {"type": "number"}


@pytest.mark.parametrize(
    ("operation", "values", "error"),
    [
        ("divide", ["1", "0"], "divide by zero"),
        ("percent_change", ["0", "10"], "old value"),
        ("subtract", ["1"], "exactly two"),
        ("add", [], "at least one"),
    ],
)
def test_calculate_rejects_invalid_input(operation, values, error):
    with pytest.raises(ValueError, match=error):
        calculate.invoke({"operation": operation, "values": values})


def test_tool_agent_can_calculate_from_document_evidence():
    llm_client = FakeToolCallingLLM(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_documents",
                        "args": {"query": "quarterly revenue"},
                        "id": "search-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate",
                        "args": {
                            "operation": "percent_change",
                            "values": [120, 150],
                        },
                        "id": "calculate-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Revenue increased by 25%."),
        ]
    )

    state = invoke_tool_agent(
        llm_client,
        FakeRetriever([make_search_result()]),
        require_evidence_tool=True,
    )

    payload = json.loads(str(tool_messages_seen_by_model(llm_client)[1].content))
    assert payload["result"] == "25"
    assert payload["unit"] == "percent"
    assert state["response"].answer == "Revenue increased by 25%."
    assert [call["name"] for call in state["response"].tool_calls] == [
        "search_documents",
        "calculate",
    ]
    assert state["response"].sources[0].chunk.id == make_search_result().chunk.id
