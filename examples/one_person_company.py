#!/usr/bin/env python3
"""A "one-person company" run by AI subagents.

You are the founder. The main Claude session acts as chief of staff and
delegates to specialist subagents (marketing, support, finance, engineering).
Each subagent has its own prompt and a restricted toolset.

Usage:
./examples/one_person_company.py "Plan the launch of a Notion template shop"
"""

import sys

import anyio

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

DEFAULT_TASK = "Plan the launch of a small digital-products shop (Notion templates) for freelancers."

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
        tools=["Write"],
        model="haiku",
    ),
    "finance": AgentDefinition(
        description="Pricing, unit economics, budget and runway estimates",
        prompt="You are the finance lead. Propose pricing tiers, estimate costs "
        "(tools, ads, fees) and break-even volume. State assumptions explicitly.",
        tools=["Read", "Write"],
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
        allowed_tools=["Task", "Read", "Write"],
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
    anyio.run(main, " ".join(sys.argv[1:]) or DEFAULT_TASK)
