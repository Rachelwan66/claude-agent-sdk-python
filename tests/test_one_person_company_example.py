"""Tests for the approval hooks in examples/one_person_company.py."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

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
    tool_names = {
        t.name
        for t in (example.sales_summary, example.send_email, example.issue_refund)
    }
    assert tool_names == {"sales_summary", "send_email", "issue_refund"}
    assert example.ops_server["name"] == "ops"
