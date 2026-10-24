#!/usr/bin/env python3
"""A "one-person company" run by AI subagents.

You are the founder. The main Claude session acts as chief of staff and
delegates to specialist subagents (marketing, support, finance, engineering).
Each subagent has its own prompt and a restricted toolset.

Business actions (emails, refunds, invoices) are exposed through an in-process
MCP server. A PreToolUse hook is the approval gate: read-only tools run freely,
side-effecting tools need the founder's confirmation, and refunds above a hard
limit are denied outright. A PostToolUse hook keeps an audit log.

Usage:
./examples/one_person_company.py "Plan the launch of a Notion template shop"
./examples/one_person_company.py --yes "..."   # auto-approve (no prompts)
"""

import sys
from typing import Any

import anyio

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

DEFAULT_TASK = "Plan the launch of a small digital-products shop (Notion templates) for freelancers."

REFUND_HARD_LIMIT_USD = 100.0
AUTO_APPROVE = False

# Tools that only read data; everything else in the ops server needs approval.
READ_ONLY_TOOLS = {"mcp__ops__sales_summary"}


def _text(msg: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": msg}], "is_error": is_error}


@tool("sales_summary", "Get last 7 days of sales (demo data)", {})
async def sales_summary(args: dict[str, Any]) -> dict[str, Any]:
    return _text("Last 7 days: 42 orders, $1,260 revenue, 3 refunds ($57).")


@tool(
    "send_email",
    "Send an email to a customer",
    {"to": str, "subject": str, "body": str},
)
async def send_email(args: dict[str, Any]) -> dict[str, Any]:
    # Demo only: replace with a real mail provider.
    return _text(f"[demo] Email sent to {args['to']}: {args['subject']}")


@tool("issue_refund", "Refund an order", {"order_id": str, "amount_usd": float})
async def issue_refund(args: dict[str, Any]) -> dict[str, Any]:
    # Demo only: replace with your payment provider.
    return _text(f"[demo] Refunded ${args['amount_usd']:.2f} for {args['order_id']}")


ops_server = create_sdk_mcp_server(
    name="ops", version="1.0.0", tools=[sales_summary, send_email, issue_refund]
)


def _deny(reason: str) -> HookJSONOutput:
    return {
        "systemMessage": f"🚫 {reason}",
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _allow(reason: str) -> HookJSONOutput:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
        },
    }


async def approval_gate(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    """Founder approval for side-effecting tools."""
    name = input_data.get("tool_name", "")
    args = input_data.get("tool_input", {})

    if name in READ_ONLY_TOOLS:
        return _allow("Read-only tool")

    if name == "mcp__ops__issue_refund" and args.get("amount_usd", 0) > REFUND_HARD_LIMIT_USD:
        return _deny(f"Refunds above ${REFUND_HARD_LIMIT_USD:.0f} must be handled manually")

    if AUTO_APPROVE:
        return _allow("Auto-approved (--yes)")

    prompt = f"\nApprove {name} {args}? [y/N] "
    answer = await anyio.to_thread.run_sync(input, prompt)
    if answer.strip().lower() in ("y", "yes"):
        return _allow("Approved by founder")
    return _deny("Founder declined")


async def audit_log(
    input_data: HookInput, tool_use_id: str | None, context: HookContext
) -> HookJSONOutput:
    print(f"[audit] {input_data.get('tool_name')} {input_data.get('tool_input')}")
    return {}


AGENTS = {
    "marketer": AgentDefinition(
        description="Positioning, content calendar, landing page copy, launch campaigns",
        prompt="You are the head of marketing for a one-person company. "
        "Produce concrete, low-budget, high-leverage plans: audience, channels, "
        "a 2-week content calendar, and landing page copy.",
        tools=["WebSearch", "Write"],
        model="sonnet",
    ),
    "support": AgentDefinition(
        description="Customer support: FAQs, reply templates, refund and escalation policy",
        prompt="You are the customer support lead. Draft an FAQ, reusable reply "
        "templates, and a simple refund/escalation policy. Keep a friendly tone.",
        tools=["Write", "mcp__ops__send_email", "mcp__ops__issue_refund"],
        model="haiku",
    ),
    "finance": AgentDefinition(
        description="Pricing, unit economics, budget and runway estimates",
        prompt="You are the finance lead. Propose pricing tiers, estimate costs "
        "(tools, ads, fees) and break-even volume. State assumptions explicitly.",
        tools=["Read", "Write", "mcp__ops__sales_summary", "mcp__ops__issue_refund"],
        model="sonnet",
    ),
    "engineer": AgentDefinition(
        description="Automation and tooling: scripts, payment/webhook wiring, site setup",
        prompt="You are the engineer. Recommend the simplest stack and automations "
        "(checkout, delivery, email, analytics) and write small scripts when needed.",
        tools=["Read", "Write", "Edit", "Bash"],
        model="sonnet",
    ),
}


async def main(task: str) -> None:
    options = ClaudeAgentOptions(
        system_prompt="You are the chief of staff of a one-person company. "
        "Break the founder's goal into parts, delegate each part to the right "
        "subagent, then merge the results into one short action plan with "
        "owners and deadlines.",
        agents=AGENTS,
        mcp_servers={"ops": ops_server},
        allowed_tools=["Task", "Read", "Write"],
        hooks={
            "PreToolUse": [HookMatcher(matcher="mcp__ops__.*", hooks=[approval_gate])],
            "PostToolUse": [HookMatcher(matcher="mcp__ops__.*", hooks=[audit_log])],
        },
        max_turns=20,
    )

    async for message in query(prompt=task, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Chief of staff: {block.text}")
        elif isinstance(message, ResultMessage) and message.total_cost_usd:
            print(f"\nCost: ${message.total_cost_usd:.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--yes" in args:
        AUTO_APPROVE = True
        args.remove("--yes")
    anyio.run(main, " ".join(args) or DEFAULT_TASK)
