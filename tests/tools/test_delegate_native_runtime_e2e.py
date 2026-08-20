from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import delegate_task
from tools.process_registry import process_registry


pytestmark = pytest.mark.integration


def _wait_for_event(
    delegation_id: str,
    event_type: str,
    timeout: float = 90.0,
):
    deadline = time.monotonic() + timeout
    deferred = []
    try:
        while time.monotonic() < deadline:
            try:
                event = process_registry.completion_queue.get(
                    timeout=min(1.0, deadline - time.monotonic())
                )
            except queue.Empty:
                continue
            if (
                event.get("delegation_id") == delegation_id
                and event.get("type") == event_type
            ):
                return event
            deferred.append(event)
    finally:
        for event in deferred:
            process_registry.completion_queue.put(event)
    raise TimeoutError(f"No {event_type} event for {delegation_id}")


def _wait_for_completion(delegation_id: str, timeout: float = 90.0):
    return _wait_for_event(delegation_id, "async_delegation", timeout)


def _write_native_fixture(repo, runtime: str) -> tuple[str, str]:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    if runtime == "claude-code":
        rule_marker = "CLAUDE_RULE_8F3D"
        skill_marker = "CLAUDE_SKILL_2A71"
        (repo / "CLAUDE.md").write_text(
            f"Always include this exact project-rule marker: {rule_marker}.\n",
            encoding="utf-8",
        )
        skill_dir = repo / ".claude" / "skills" / "hermes-e2e-probe"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: hermes-e2e-probe\n"
            "description: Use when asked for the Hermes native probe marker.\n"
            "---\n"
            f"Return this exact skill marker: {skill_marker}.\n",
            encoding="utf-8",
        )
        return rule_marker, skill_marker
    if runtime == "codex":
        rule_marker = "CODEX_RULE_6C4E"
        skill_marker = "CODEX_SKILL_1D92"
        (repo / "AGENTS.md").write_text(
            f"Always include this exact project-rule marker: {rule_marker}.\n",
            encoding="utf-8",
        )
        skill_dir = repo / ".agents" / "skills" / "hermes-e2e-probe"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: hermes-e2e-probe\n"
            "description: Use when asked for the Hermes native probe marker.\n"
            "---\n"
            f"Return this exact skill marker: {skill_marker}.\n",
            encoding="utf-8",
        )
        return rule_marker, skill_marker
    raise AssertionError(f"Unsupported fixture runtime: {runtime}")


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_inherits_claude_rules_and_skills(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI is not installed")

    repo = tmp_path / "claude-native-delegate"
    repo.mkdir()
    expected = _write_native_fixture(repo, "claude-code")
    parent = _make_mock_parent(depth=0)
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )

    result = json.loads(
        delegate_task(
            goal=(
                "Invoke the hermes-e2e-probe skill. Return exactly the project-rule "
                "marker and skill marker separated by one space, with no other text."
            ),
            runtime="claude-code",
            parent_agent=parent,
            background=False,
        )
    )

    entry = result["results"][0]
    assert entry["summary"].strip() == " ".join(expected)
    assert entry["status"] == "completed"
    assert entry["exit_reason"] == "completed"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_inherits_codex_rules_and_skills(tmp_path, monkeypatch):
    if shutil.which("codex") is None:
        pytest.skip("Codex CLI is not installed")

    repo = tmp_path / "codex-native-delegate"
    repo.mkdir()
    expected = _write_native_fixture(repo, "codex")
    parent = _make_mock_parent(depth=0)
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )

    result = json.loads(
        delegate_task(
            goal=(
                "Invoke the hermes-e2e-probe skill. Return exactly the project-rule "
                "marker and skill marker separated by one space, with no other text."
            ),
            runtime="codex",
            parent_agent=parent,
            background=False,
        )
    )

    entry = result["results"][0]
    assert entry["summary"].strip() == " ".join(expected)
    assert entry["status"] == "completed"
    assert entry["exit_reason"] == "completed"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
@pytest.mark.parametrize("runtime", ["claude-code", "codex"])
def test_public_delegate_task_honors_managed_native_configuration(
    runtime,
    tmp_path,
    monkeypatch,
):
    binary = "claude" if runtime == "claude-code" else "codex"
    if shutil.which(binary) is None:
        pytest.skip(f"{runtime} CLI is not installed")
    if runtime == "claude-code":
        pytest.importorskip("claude_agent_sdk")

    env_prefix = "CLAUDE" if runtime == "claude-code" else "CODEX"
    model = os.environ.get(f"HERMES_NATIVE_DELEGATION_{env_prefix}_MODEL")
    if not model:
        pytest.skip(
            f"set HERMES_NATIVE_DELEGATION_{env_prefix}_MODEL to a provider-valid model"
        )
    effort = os.environ.get(f"HERMES_NATIVE_DELEGATION_{env_prefix}_EFFORT", "xhigh")
    approval_mode = "auto" if runtime == "claude-code" else "approve_for_me"

    repo = tmp_path / f"{runtime}-native-config"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    parent = _make_mock_parent(depth=0)
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._native_classifier_approvals_enabled",
        lambda: True,
    )

    payload = json.loads(
        delegate_task(
            goal="Return exactly NATIVE_CONFIG_OK with no other text.",
            runtime=runtime,
            native={
                "model": model,
                "effort": effort,
                "approval_mode": approval_mode,
            },
            parent_agent=parent,
            background=False,
        )
    )
    assert "error" not in payload, payload
    entry = payload["results"][0]

    assert entry["summary"].strip() == "NATIVE_CONFIG_OK"
    assert entry["status"] == "completed"
    assert entry["native_model_requested"] == model
    assert entry["native_effort_requested"] == effort
    assert entry["native_approval_mode_requested"] == approval_mode
    assert entry["native_model_resolved"]
    if runtime == "codex":
        assert "native_effort_resolved" not in entry
        assert entry["native_approval_policy_resolved"] == "on-request"
        assert entry["native_approvals_reviewer_resolved"] == "auto_review"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
@pytest.mark.parametrize("runtime", ["claude-code", "codex"])
def test_public_delegate_task_answers_native_question(
    runtime,
    tmp_path,
    monkeypatch,
):
    binary = "claude" if runtime == "claude-code" else "codex"
    if shutil.which(binary) is None:
        pytest.skip(f"{runtime} CLI is not installed")
    if runtime == "claude-code":
        pytest.importorskip("claude_agent_sdk")

    repo = tmp_path / f"{runtime}-native-question"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    parent = _make_mock_parent(depth=0)
    parent.session_id = f"{runtime}-question-e2e-parent"
    parent._current_task_id = None
    question_tool = "AskUserQuestion" if runtime == "claude-code" else "ask_user"
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )

    dispatch = json.loads(
        delegate_task(
            goal=(
                f"Before answering, call the {question_tool} tool to ask "
                "exactly 'Which marker should I return?' with choices ALPHA and BETA. "
                "After receiving the answer, return exactly the selected marker and "
                "no other text."
            ),
            runtime=runtime,
            parent_agent=parent,
            background=True,
        )
    )
    subagent_id = dispatch["subagent_ids"][0]
    input_event = _wait_for_event(
        dispatch["delegation_id"],
        "async_delegation_input",
    )
    request = input_event["input_request"]
    question = request["questions"][0]
    assert question["question"] == "Which marker should I return?"

    responded = json.loads(
        delegate_task(
            action="respond",
            runtime=runtime,
            subagent_id=subagent_id,
            request_id=request["request_id"],
            answers={question["id"]: ["ALPHA"]},
            parent_agent=parent,
        )
    )
    assert responded["status"] == "answered"

    completion = _wait_for_completion(dispatch["delegation_id"])
    assert completion["results"][0]["summary"].strip() == "ALPHA"
    assert completion["results"][0]["exit_reason"] == "completed"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_steers_active_claude_worker(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI is not installed")

    repo = tmp_path / "claude-native-steer"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"permissions": {"ask": ["Bash(sleep *)"]}}),
        encoding="utf-8",
    )
    approval_requested = threading.Event()

    def approve(command, description, **kwargs):
        if kwargs.get("tool_name") == "Bash":
            approval_requested.set()
        return "once"

    parent = _make_mock_parent(depth=0)
    parent.session_id = "native-steer-e2e-parent"
    parent._current_task_id = None
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._get_subagent_approval_callback",
        lambda: approve,
    )

    dispatch = json.loads(
        delegate_task(
            goal=(
                "Use Bash to run `sleep 6`. After the command finishes, return "
                "exactly ORIGINAL_RESPONSE and no other text."
            ),
            runtime="claude-code",
            parent_agent=parent,
            background=True,
        )
    )
    assert dispatch["status"] == "dispatched"
    subagent_id = dispatch["subagent_ids"][0]

    try:
        assert approval_requested.wait(timeout=30), "Claude never requested Bash approval"
        steer = json.loads(
            delegate_task(
                action="steer",
                subagent_id=subagent_id,
                message="Return exactly STEERED_RESPONSE and no other text.",
                parent_agent=parent,
            )
        )
        assert steer["status"] == "queued"
        completion = _wait_for_completion(dispatch["delegation_id"])
    finally:
        delegate_task(action="stop", subagent_id=subagent_id, parent_agent=parent)

    assert completion["status"] == "completed"
    assert completion["results"][0]["summary"].strip() == "STEERED_RESPONSE"
    assert completion["results"][0]["exit_reason"] == "completed"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_stops_active_claude_worker(tmp_path, monkeypatch):
    pytest.importorskip("claude_agent_sdk")
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI is not installed")

    repo = tmp_path / "claude-native-stop"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    claude_dir = repo / ".claude"
    claude_dir.mkdir()
    (claude_dir / "settings.local.json").write_text(
        json.dumps({"permissions": {"ask": ["Bash(sleep *)"]}}),
        encoding="utf-8",
    )
    approval_requested = threading.Event()

    def approve(command, description, **kwargs):
        if kwargs.get("tool_name") == "Bash":
            approval_requested.set()
        return "once"

    parent = _make_mock_parent(depth=0)
    parent.session_id = "native-stop-e2e-parent"
    parent._current_task_id = None
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )
    monkeypatch.setattr(
        "tools.delegate_tool._get_subagent_approval_callback",
        lambda: approve,
    )

    dispatch = json.loads(
        delegate_task(
            goal=(
                "Use Bash to run `sleep 30`. After the command finishes, return "
                "exactly SHOULD_NOT_FINISH and no other text."
            ),
            runtime="claude-code",
            parent_agent=parent,
            background=True,
        )
    )
    assert dispatch["status"] == "dispatched"
    subagent_id = dispatch["subagent_ids"][0]
    assert approval_requested.wait(timeout=30), "Claude never requested Bash approval"

    stopped = json.loads(
        delegate_task(
            action="stop",
            subagent_id=subagent_id,
            parent_agent=parent,
        )
    )
    assert stopped["status"] == "interrupt_requested"
    completion = _wait_for_completion(dispatch["delegation_id"])

    assert completion["results"][0]["exit_reason"] == "interrupted"
    live = json.loads(delegate_task(action="list", parent_agent=parent))
    assert live["count"] == 0


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_steers_active_codex_worker(tmp_path, monkeypatch):
    if shutil.which("codex") is None:
        pytest.skip("Codex CLI is not installed")

    repo = tmp_path / "codex-native-steer"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "AGENTS.md").write_text(
        "For the codex-steer-probe, first run `sleep 6` in the shell. "
        "Then follow the latest user instruction exactly.\n",
        encoding="utf-8",
    )
    parent = _make_mock_parent(depth=0)
    parent.session_id = "codex-steer-e2e-parent"
    parent._current_task_id = None
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )

    dispatch = json.loads(
        delegate_task(
            goal=(
                "Run the codex-steer-probe. After the shell command finishes, "
                "return exactly CODEX_ORIGINAL and no other text."
            ),
            runtime="codex",
            parent_agent=parent,
            background=True,
        )
    )
    assert dispatch["status"] == "dispatched"
    subagent_id = dispatch["subagent_ids"][0]

    queued = None
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = json.loads(
            delegate_task(
                action="steer",
                subagent_id=subagent_id,
                message="Return exactly CODEX_STEERED and no other text.",
                parent_agent=parent,
            )
        )
        if response.get("status") == "queued":
            queued = response
            break
        time.sleep(0.2)
    assert queued is not None, "Codex never exposed an active turn for steering"

    completion = _wait_for_completion(dispatch["delegation_id"])
    assert completion["status"] == "completed"
    assert completion["results"][0]["summary"].strip() == "CODEX_STEERED"
    assert completion["results"][0]["exit_reason"] == "completed"


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_resumes_claude_session(tmp_path):
    if shutil.which("claude") is None:
        pytest.skip("claude is not installed")
    pytest.importorskip("claude_agent_sdk")

    repo = tmp_path / "claude-native-resume"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    parent = _make_mock_parent(depth=0)
    token = "CLAUDE_RESUME_E2E_7B29"

    with patch(
        "tools.delegate_tool._resolve_workspace_hint",
        return_value=str(repo),
    ):
        first = json.loads(
            delegate_task(
                goal=(
                    f"Remember the token {token} for this native session. "
                    "Return exactly STORED."
                ),
                runtime="claude-code",
                parent_agent=parent,
            )
        )
        native_session_id = first["results"][0]["native_session_id"]
        resumed = json.loads(
            delegate_task(
                goal="Return exactly the token I asked you to remember earlier.",
                runtime="claude-code",
                resume_session_id=native_session_id,
                parent_agent=parent,
            )
        )

    assert resumed["results"][0]["summary"].strip() == token


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_resumes_codex_session(tmp_path):
    if shutil.which("codex") is None:
        pytest.skip("Codex CLI is not installed")

    repo = tmp_path / "codex-native-resume"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    parent = _make_mock_parent(depth=0)
    token = "CODEX_RESUME_E2E_4D81"

    with patch(
        "tools.delegate_tool._resolve_workspace_hint",
        return_value=str(repo),
    ):
        first = json.loads(
            delegate_task(
                goal=(
                    f"Remember the token {token} for this native thread. "
                    "Return exactly STORED."
                ),
                runtime="codex",
                parent_agent=parent,
            )
        )
        native_session_id = first["results"][0]["native_session_id"]
        resumed = json.loads(
            delegate_task(
                goal="Return exactly the token I asked you to remember earlier.",
                runtime="codex",
                resume_session_id=native_session_id,
                parent_agent=parent,
            )
        )

    assert resumed["results"][0]["summary"].strip() == token


@pytest.mark.skipif(
    os.environ.get("HERMES_RUN_NATIVE_DELEGATION_E2E") != "1",
    reason="set HERMES_RUN_NATIVE_DELEGATION_E2E=1 for paid provider E2E",
)
def test_public_delegate_task_stops_active_codex_worker(tmp_path, monkeypatch):
    if shutil.which("codex") is None:
        pytest.skip("Codex CLI is not installed")

    repo = tmp_path / "codex-native-stop"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "AGENTS.md").write_text(
        "For the codex-stop-probe, first run `sleep 30` in the shell. "
        "Do not answer before the command finishes.\n",
        encoding="utf-8",
    )
    parent = _make_mock_parent(depth=0)
    parent.session_id = "codex-stop-e2e-parent"
    parent._current_task_id = None
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_workspace_hint",
        lambda _parent: str(repo),
    )

    dispatch = json.loads(
        delegate_task(
            goal="Run the codex-stop-probe, then return exactly SHOULD_NOT_FINISH.",
            runtime="codex",
            parent_agent=parent,
            background=True,
        )
    )
    assert dispatch["status"] == "dispatched"
    subagent_id = dispatch["subagent_ids"][0]

    active = False
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        response = json.loads(
            delegate_task(
                action="steer",
                subagent_id=subagent_id,
                message="Continue the current probe.",
                parent_agent=parent,
            )
        )
        if response.get("status") == "queued":
            active = True
            break
        time.sleep(0.2)
    assert active, "Codex never exposed an active turn for interruption"

    stopped = json.loads(
        delegate_task(
            action="stop",
            subagent_id=subagent_id,
            parent_agent=parent,
        )
    )
    assert stopped["status"] == "interrupt_requested"
    completion = _wait_for_completion(dispatch["delegation_id"])

    assert completion["results"][0]["exit_reason"] == "interrupted"
    live = json.loads(delegate_task(action="list", parent_agent=parent))
    assert live["count"] == 0
