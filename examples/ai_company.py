#!/usr/bin/env python3
"""Example: an "AI company" built from a CEO agent and department sub-agents.

The main session acts as the CEO. It does no hands-on work itself; it delegates
to four sub-agents (product manager, engineer, QA, marketing), each with its own
prompt, model, and least-privilege tool list. The pieces map to SDK features:

- Org chart        -> ``agents`` (``AgentDefinition`` per role)
- Company systems  -> an in-process MCP server (a shared ticket board)
- Company policy   -> a ``PreToolUse`` hook that blocks risky shell commands
                      and file writes outside the workspace
- Budget           -> ``max_budget_usd`` / ``max_turns``

Departments do not see each other's conversations, so they hand work off
through files in the workspace and through the ticket board.

Note: Bash is pre-approved so the engineer and QA can run tests. The policy hook
is a simple blocklist, so run this on a machine or container you don't mind an
agent executing commands in, and tighten BLOCKED_COMMANDS for your setup.

Usage:
    python examples/ai_company.py
    python examples/ai_company.py "Build a CLI that converts CSV to JSON"
    python examples/ai_company.py --workspace ./my_company --budget 2.0 "..."
"""

import argparse
from pathlib import Path
from typing import Any

import anyio

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookCallback,
    HookContext,
    HookInput,
    HookJSONOutput,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

DEFAULT_GOAL = "Ship a small Python CLI todo app with add, list, and done commands."

# Names of the built-in tool the CEO uses to delegate to a sub-agent.
# Older CLI versions call it "Task"; newer ones call it "Agent".
DELEGATION_TOOLS = {"Task", "Agent"}


# ---------------------------------------------------------------------------
# Company systems: a shared ticket board exposed as in-process MCP tools.
# The data lives in this Python process, so your app can persist or display it.
# ---------------------------------------------------------------------------

TICKETS: list[dict[str, Any]] = []
TICKET_STATUSES = {"open", "in_progress", "done", "rejected"}


def _text(message: str, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"content": [{"type": "text", "text": message}]}
    if is_error:
        result["is_error"] = True
    return result


@tool(
    "create_ticket",
    "Create a work ticket assigned to a department",
    {"title": str, "owner": str},
)
async def create_ticket(args: dict[str, Any]) -> dict[str, Any]:
    ticket = {
        "id": len(TICKETS) + 1,
        "title": args["title"],
        "owner": args["owner"],
        "status": "open",
        "note": "",
    }
    TICKETS.append(ticket)
    return _text(
        f"Created ticket #{ticket['id']}: {ticket['title']} -> {ticket['owner']}"
    )


@tool(
    "update_ticket",
    "Update a ticket's status (open, in_progress, done, rejected) with a short note",
    {"id": int, "status": str, "note": str},
)
async def update_ticket(args: dict[str, Any]) -> dict[str, Any]:
    if args["status"] not in TICKET_STATUSES:
        return _text(
            f"Invalid status {args['status']!r}; use one of {sorted(TICKET_STATUSES)}",
            is_error=True,
        )
    for ticket in TICKETS:
        if ticket["id"] == args["id"]:
            ticket["status"] = args["status"]
            ticket["note"] = args["note"]
            return _text(f"Ticket #{ticket['id']} is now {ticket['status']}")
    return _text(f"No ticket with id {args['id']}", is_error=True)


@tool("list_tickets", "List every ticket on the board", {})
async def list_tickets(args: dict[str, Any]) -> dict[str, Any]:
    if not TICKETS:
        return _text("The board is empty.")
    return _text("\n".join(_format_ticket(t) for t in TICKETS))


def _format_ticket(ticket: dict[str, Any]) -> str:
    line = f"#{ticket['id']} [{ticket['status']}] {ticket['title']} (owner: {ticket['owner']})"
    return f"{line} - {ticket['note']}" if ticket["note"] else line


BOARD = create_sdk_mcp_server(
    name="board", version="1.0.0", tools=[create_ticket, update_ticket, list_tickets]
)
BOARD_READ = ["mcp__board__list_tickets"]
BOARD_WRITE = ["mcp__board__create_ticket", "mcp__board__update_ticket"]
BOARD_ALL = BOARD_READ + BOARD_WRITE


# ---------------------------------------------------------------------------
# Org chart: one AgentDefinition per department. The CEO picks a department
# by its description, so write descriptions like job postings.
# ---------------------------------------------------------------------------

AGENTS: dict[str, AgentDefinition] = {
    "product-manager": AgentDefinition(
        description=(
            "Turns a business goal into a written spec with user stories and "
            "acceptance criteria. Use first, before any engineering work."
        ),
        prompt=(
            "You are the product manager. Write a concise spec to docs/spec.md with "
            "user stories and testable acceptance criteria. Keep scope small. "
            "Never write code. When done, mark your ticket done on the board."
        ),
        tools=["Read", "Write", "Glob", *BOARD_ALL],
        model="sonnet",
    ),
    "engineer": AgentDefinition(
        description=(
            "Implements docs/spec.md as working code with automated tests, or fixes "
            "defects reported by QA."
        ),
        prompt=(
            "You are a senior engineer. Implement exactly what docs/spec.md asks for, "
            "with tests. Run the tests and make them pass before you finish. If QA "
            "filed defects in docs/qa_report.md, fix those. Update your ticket on the "
            "board when done."
        ),
        tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep", *BOARD_ALL],
        model="opus",
    ),
    "qa": AgentDefinition(
        description=(
            "Reviews the implementation against docs/spec.md, runs the tests, and "
            "reports PASS or a list of defects."
        ),
        prompt=(
            "You are QA. You may not edit source code. Check every acceptance "
            "criterion in docs/spec.md, run the tests, and write docs/qa_report.md. "
            "Its first line must be exactly 'VERDICT: PASS' or 'VERDICT: FAIL', "
            "followed by numbered defects with file:line. Update your ticket."
        ),
        tools=["Read", "Write", "Bash", "Glob", "Grep", *BOARD_ALL],
        model="sonnet",
    ),
    "marketing": AgentDefinition(
        description=(
            "Writes launch copy and release notes once QA has passed the product."
        ),
        prompt=(
            "You are marketing. Read docs/spec.md and the code, then write "
            "docs/launch.md: a one-line pitch, three key features, and a quick-start "
            "example. Plain, concrete language; no claims the product can't back up. "
            "Update your ticket."
        ),
        tools=["Read", "Write", "Glob", *BOARD_ALL],
        model="haiku",
    ),
}

CEO_PROMPT = """\
You are the CEO of a small software company. You never do hands-on work yourself:
you plan, delegate to your departments, and report.

Your departments: product-manager, engineer, qa, marketing.

For every goal, follow this process:
1. Create one ticket per department on the board (mcp__board__create_ticket).
2. Have product-manager write docs/spec.md.
3. Have engineer implement it.
4. Have qa review it. If docs/qa_report.md says VERDICT: FAIL, send the defects back
   to engineer and then to qa again. Stop after 3 rounds even if QA still fails.
5. Only after VERDICT: PASS, have marketing write docs/launch.md.
6. Call mcp__board__list_tickets and finish with a short status report: what
   shipped, the final QA verdict, and anything left open.

Departments cannot see each other's conversations. When you delegate, tell each
department exactly which files to read and write.
"""


# ---------------------------------------------------------------------------
# Company policy: a PreToolUse hook that blocks risky actions for everyone,
# including sub-agents.
# ---------------------------------------------------------------------------

BLOCKED_COMMANDS = ("git push", "rm -rf", "sudo ", "curl ", "wget ")


def make_policy_hook(workspace: Path) -> HookCallback:
    """Build a hook that keeps file writes inside the workspace and blocks risky commands."""
    root = workspace.resolve()

    async def company_policy(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        data: dict[str, Any] = dict(input_data)
        tool_name = str(data.get("tool_name", ""))
        tool_input: dict[str, Any] = data.get("tool_input") or {}

        reason = None
        if tool_name == "Bash":
            command = tool_input.get("command", "")
            for pattern in BLOCKED_COMMANDS:
                if pattern in command:
                    reason = f"Company policy forbids '{pattern.strip()}'."
                    break
        elif tool_name in {"Write", "Edit", "MultiEdit", "NotebookEdit"}:
            raw_path = tool_input.get("file_path") or tool_input.get("notebook_path")
            if raw_path:
                target = Path(raw_path)
                if not target.is_absolute():
                    target = root / target
                if not target.resolve().is_relative_to(root):
                    reason = f"Company policy: files must stay inside {root}."

        if reason is None:
            return {}
        print(f"  [policy] blocked {tool_name}: {reason}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return company_policy


# ---------------------------------------------------------------------------
# Running the company
# ---------------------------------------------------------------------------


def build_options(workspace: Path, budget: float, max_turns: int) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=CEO_PROMPT,
        agents=AGENTS,
        mcp_servers={"board": BOARD},
        # acceptEdits auto-approves file edits but not shell commands; the engineer
        # and QA need Bash to run tests. Every command still passes through the
        # policy hook below, which denies the blocked patterns.
        allowed_tools=[*BOARD_ALL, "Bash"],
        cwd=str(workspace),
        permission_mode="acceptEdits",
        hooks={
            "PreToolUse": [
                HookMatcher(matcher=None, hooks=[make_policy_hook(workspace)])
            ]
        },
        max_turns=max_turns,
        max_budget_usd=budget,
    )


def _summarize_input(block: ToolUseBlock) -> str:
    for key in ("description", "file_path", "command", "title", "pattern"):
        value = block.input.get(key)
        if value:
            text = str(value).replace("\n", " ")
            return text if len(text) <= 80 else text[:77] + "..."
    return ""


async def run_company(
    goal: str, workspace: Path, budget: float, max_turns: int
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    options = build_options(workspace, budget, max_turns)

    # Maps the CEO's delegation tool_use_id to the department it went to, so
    # messages from inside a sub-agent (which carry parent_tool_use_id) can be
    # labelled with the right department.
    departments: dict[str, str] = {}

    print(f"Goal: {goal}")
    print(f"Workspace: {workspace.resolve()}\n")

    async with ClaudeSDKClient(options=options) as ceo:
        await ceo.query(f"Goal: {goal}")
        async for message in ceo.receive_response():
            if isinstance(message, AssistantMessage):
                speaker = "CEO"
                if message.parent_tool_use_id:
                    speaker = departments.get(message.parent_tool_use_id, "department")
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        print(f"[{speaker}] {block.text.strip()}\n")
                    elif isinstance(block, ToolUseBlock):
                        if block.name in DELEGATION_TOOLS:
                            dept = str(block.input.get("subagent_type", "department"))
                            departments[block.id] = dept
                            print(
                                f"[{speaker}] => delegates to {dept}: {_summarize_input(block)}"
                            )
                        else:
                            print(
                                f"[{speaker}] -> {block.name} {_summarize_input(block)}"
                            )
            elif isinstance(message, ResultMessage):
                print("\n=== Ticket board ===")
                for ticket in TICKETS:
                    print(_format_ticket(ticket))
                if not TICKETS:
                    print("(no tickets)")
                cost = message.total_cost_usd or 0.0
                print(
                    f"\nFinished ({message.subtype}) in {message.num_turns} turns, cost ${cost:.4f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AI company on a goal.")
    parser.add_argument(
        "goal", nargs="?", default=DEFAULT_GOAL, help="What the company should ship"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("company_workspace"),
        help="Directory the company works in (default: ./company_workspace)",
    )
    parser.add_argument(
        "--budget", type=float, default=5.0, help="Max spend in USD (default: 5.0)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=80, help="Max CEO turns (default: 80)"
    )
    args = parser.parse_args()

    anyio.run(run_company, args.goal, args.workspace, args.budget, args.max_turns)


if __name__ == "__main__":
    main()
