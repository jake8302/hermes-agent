"""Claude Agent SDK early-return turns must reach SessionDB exactly once."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("claude_agent_sdk")

from agent.claude_agent_sdk_runtime import ClaudeTurnResult, run_claude_agent_sdk_turn
from hermes_state import SessionDB
from run_agent import AIAgent


def _turn() -> ClaudeTurnResult:
    return ClaudeTurnResult(
        final_text="CLAUDE_ASSISTANT",
        session_id="claude-session-persist",
        projected_messages=[
            {"role": "assistant", "content": "CLAUDE_ASSISTANT"}
        ],
        num_turns=1,
    )


def test_claude_success_flushes_and_reports_persisted():
    agent = MagicMock()
    agent._claude_sdk_session = MagicMock()
    agent._claude_sdk_session.run_turn.return_value = _turn()
    agent._session_db = None
    agent._interrupt_requested = False
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    messages = [{"role": "user", "content": "hello"}]

    result = run_claude_agent_sdk_turn(
        agent,
        user_message="hello",
        original_user_message="hello",
        messages=messages,
        effective_task_id="task-1",
    )

    assert result["completed"] is True
    assert result["agent_persisted"] is True
    assert isinstance(result["messages"][-1]["timestamp"], float)


def test_claude_turn_persists_each_message_exactly_once(tmp_path: Path):
    db = SessionDB(tmp_path / "state.db")
    session_id = "sess-claude-once"
    db.create_session(session_id=session_id, source="telegram", model="claude-code")
    agent = AIAgent(
        api_key="test-key",
        base_url="http://localhost/claude-agent-sdk",
        provider="claude-code",
        api_mode="claude_agent_sdk",
        model="claude-code",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        session_db=db,
        session_id=session_id,
    )
    agent._session_db_created = True
    native_session = MagicMock()
    native_session.run_turn.return_value = _turn()
    setattr(agent, "_claude_sdk_session", native_session)

    user_message = {"role": "user", "content": "CLAUDE_USER_TURN"}
    messages = [user_message]
    agent._flush_messages_to_session_db(messages)

    result = run_claude_agent_sdk_turn(
        agent,
        user_message="CLAUDE_USER_TURN",
        original_user_message="CLAUDE_USER_TURN",
        messages=messages,
        effective_task_id="task-1",
    )

    assert result["agent_persisted"] is True
    rows = db.get_messages(session_id, include_inactive=True)
    contents = [row["content"] for row in rows]
    assert contents.count("CLAUDE_USER_TURN") == 1, contents
    assert contents.count("CLAUDE_ASSISTANT") == 1, contents
    assistant_row = next(
        row for row in rows if row["content"] == "CLAUDE_ASSISTANT"
    )
    assert isinstance(assistant_row["timestamp"], float)
    hits = {row["session_id"] for row in db.search_messages("CLAUDE_ASSISTANT")}
    assert session_id in hits
