"""Codex app-server session lifecycle on hard agent teardown (#65260).

The Codex runtime drops ``agent._codex_session`` on turn crash and on
retirement (agent/codex_runtime.py), but ``AIAgent.close()`` — the hard
teardown for /new, /reset, and session expiry — had no owner for it, so
the app-server child process survived until interpreter exit.
"""

import threading
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class _FakeCodexSession:
    def __init__(self, raises: bool = False):
        self.close_calls = 0
        self._raises = raises

    def close(self):
        self.close_calls += 1
        if self._raises:
            raise RuntimeError("app-server already dead")

    def request_steer(self, text: str) -> bool:
        self.steer_text = text
        return True


def _bare_agent(session_id: str) -> AIAgent:
    """Minimal agent shell exercising close() without a real build."""
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.client = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._end_session_on_close = False
    agent._session_messages = ["retained"]
    return agent


def test_agent_close_releases_codex_app_server_session(monkeypatch):
    agent = _bare_agent("test-codex-lifecycle")
    codex_session = _FakeCodexSession()
    agent._codex_session = codex_session

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()
    agent.close()

    # Idempotent: the second close must not re-close a released session.
    assert codex_session.close_calls == 1
    assert agent._codex_session is None
    assert agent._session_messages == []


def test_agent_steer_reaches_active_codex_app_server_session():
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "codex_app_server"
    codex_session = _FakeCodexSession()
    agent._codex_session = codex_session

    assert agent.steer("focus on the parser") is True
    assert codex_session.steer_text == "focus on the parser"


def test_agent_stamps_resume_id_on_codex_app_server_session():
    from agent.codex_runtime import run_codex_app_server_turn

    agent = MagicMock()
    agent._codex_session = None
    agent.session_cwd = "/tmp/native-codex-resume"
    agent._native_resume_session_id = "thread-existing-42"
    agent._interrupt_requested = False

    with patch(
        "agent.transports.codex_app_server_session.CodexAppServerSession"
    ) as Session:
        Session.return_value.run_turn.side_effect = RuntimeError("stop after init")
        run_codex_app_server_turn(
            agent,
            user_message="resume",
            original_user_message="resume",
            messages=[],
            effective_task_id="task-1",
        )

    assert Session.call_args.kwargs["resume_thread_id"] == "thread-existing-42"


def test_close_clears_reference_even_when_session_close_raises(monkeypatch):
    """A wedged app-server must not strand a stale session reference.

    The attribute is cleared BEFORE close() precisely so a raising close
    can't leave a dead session attached to the agent.
    """
    agent = _bare_agent("test-codex-lifecycle-raises")
    codex_session = _FakeCodexSession(raises=True)
    agent._codex_session = codex_session

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()

    assert codex_session.close_calls == 1
    assert agent._codex_session is None


def test_close_without_codex_session_is_a_noop(monkeypatch):
    """Non-Codex sessions (the common case) must be unaffected."""
    agent = _bare_agent("test-no-codex")

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()

    assert getattr(agent, "_codex_session", None) is None
