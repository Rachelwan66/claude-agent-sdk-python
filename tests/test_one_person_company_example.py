"""Tests for the approval hooks in examples/one_person_company.py."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest

EXAMPLE = Path(__file__).parent.parent / "examples" / "one_person_company.py"


@pytest.fixture
def example() -> Any:
    spec = importlib.util.spec_from_file_location("one_person_company", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decision(result: dict[str, Any]) -> str:
    return result["hookSpecificOutput"]["permissionDecision"]


async def _gate(
    example: Any, tool_name: str, tool_input: dict[str, Any]
) -> dict[str, Any]:
    return await example.approval_gate(
        {"tool_name": tool_name, "tool_input": tool_input}, None, {"signal": None}
    )


@pytest.mark.asyncio
async def test_read_only_tool_is_allowed_without_prompt(example, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("should not prompt"))
    assert _decision(await _gate(example, "mcp__ops__sales_summary", {})) == "allow"


@pytest.mark.asyncio
async def test_refund_above_limit_is_denied_even_with_auto_approve(example):
    example.AUTO_APPROVE = True
    result = await _gate(example, "mcp__ops__issue_refund", {"amount_usd": 500})
    assert _decision(result) == "deny"


@pytest.mark.asyncio
async def test_auto_approve_allows_side_effects(example):
    example.AUTO_APPROVE = True
    result = await _gate(example, "mcp__ops__send_email", {"to": "a@b.c"})
    assert _decision(result) == "allow"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "amount", [None, -1, 0, float("nan"), float("inf"), "20", True]
)
async def test_invalid_refund_amount_is_denied(example, monkeypatch, amount):
    example.AUTO_APPROVE = True
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("should not prompt"))
    result = await _gate(example, "mcp__ops__issue_refund", {"amount_usd": amount})
    assert _decision(result) == "deny"


@pytest.mark.asyncio
async def test_refund_at_limit_is_allowed(example):
    example.AUTO_APPROVE = True
    result = await _gate(example, "mcp__ops__issue_refund", {"amount_usd": 100})
    assert _decision(result) == "allow"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "expected"),
    [("y", "allow"), ("YES", "allow"), ("n", "deny"), ("", "deny")],
)
async def test_founder_prompt_decides(example, monkeypatch, answer, expected):
    example.AUTO_APPROVE = False
    monkeypatch.setattr("builtins.input", lambda _: answer)
    result = await _gate(example, "mcp__ops__issue_refund", {"amount_usd": 20})
    assert _decision(result) == expected


@pytest.mark.asyncio
async def test_ops_server_exposes_all_tools(example):
    server = example.ops_server["instance"]
    response = await server.request_handlers[ListToolsRequest](
        ListToolsRequest(method="tools/list")
    )
    tool_names = {t.name for t in response.root.tools}
    assert tool_names == {"sales_summary", "send_email", "issue_refund"}
    assert example.ops_server["name"] == "ops"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("sales_summary", {}, "42 orders"),
        (
            "send_email",
            {"to": "test@example.com", "subject": "Test", "body": "Demo"},
            "[demo] Email sent",
        ),
        (
            "issue_refund",
            {"order_id": "test-order", "amount_usd": 20},
            "[demo] Refunded $20.00",
        ),
    ],
)
async def test_ops_tools_can_be_called_through_mcp(example, name, arguments, expected):
    server = example.ops_server["instance"]
    response = await server.request_handlers[CallToolRequest](
        CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=name, arguments=arguments),
        )
    )
    assert not response.root.isError
    assert expected in response.root.content[0].text
