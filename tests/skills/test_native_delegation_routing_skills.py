"""Instruction contracts for provider-native delegation routing."""

from pathlib import Path
import re

import pytest

REPO = Path(__file__).resolve().parents[2]
SKILLS = REPO / "skills"


@pytest.mark.parametrize(
    ("relative_path", "runtime", "standalone_commands", "unqualified_rule"),
    [
        (
            "autonomous-ai-agents/claude-code/SKILL.md",
            "claude-code",
            ("`claude -p`", "tmux"),
            "**Prefer print mode (`-p`) for single tasks**",
        ),
        (
            "autonomous-ai-agents/codex/SKILL.md",
            "codex",
            ("`codex exec`", "`codex review`"),
            "**Use `exec` for one-shots**",
        ),
    ],
)
def test_provider_skill_routes_managed_subagents_through_delegate_task(
    relative_path: str,
    runtime: str,
    standalone_commands: tuple[str, ...],
    unqualified_rule: str,
) -> None:
    text = (SKILLS / relative_path).read_text(encoding="utf-8")
    routing = text.split("## Routing precedence", 1)[1].split("## Prerequisites", 1)[0]
    routing_normalized = " ".join(routing.split())

    assert f'delegate_task(runtime="{runtime}"' in routing
    assert "Hermes responsible for worker identity" in routing
    assert "waiting_for_input" in routing
    assert "clarify" in routing
    assert 'action="respond"' in routing
    assert "Secret-input requests fail closed" in routing_normalized
    assert "standalone CLI" in routing
    for command in standalone_commands:
        assert command in routing

    assert "## Rules for standalone" in text
    assert unqualified_rule not in text
    before_reminder, reminder = text.rsplit("## Routing reminder", 1)
    assert before_reminder
    assert f'delegate_task(runtime="{runtime}"' in reminder
    assert "explicitly asks" in reminder
    assert "\n## " not in reminder


def test_review_skill_preserves_explicit_native_runtime() -> None:
    text = (
        SKILLS
        / "software-development"
        / "requesting-code-review"
        / "SKILL.md"
    ).read_text(encoding="utf-8")
    routing = text.split("## Step 5 — Independent reviewer subagent", 1)[1].split(
        "The reviewer gets ONLY the diff", 1
    )[0]

    assert 'runtime="claude-code"' in routing
    assert 'runtime="codex"' in routing
    assert 'Otherwise use `runtime="hermes"`' in routing
    assert "Never replace" in routing


def test_all_bundled_delegate_examples_choose_runtime_explicitly() -> None:
    missing: list[str] = []
    for path in sorted(SKILLS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"delegate_task\s*\(", text):
            tail = text[match.end() :]
            first_line = tail.split("\n", 1)[0]
            if ")" in first_line:
                block = first_line.split(")", 1)[0]
            else:
                end = tail.find("\n)")
                block = tail[: end if end >= 0 else 1200]
            if "runtime=" not in block:
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{path.relative_to(SKILLS)}:{line}")

    assert missing == []
