from __future__ import annotations

from collections import deque
import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("claude_agent_sdk")
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)


def _result(text: str, session_id: str = "claude-session-1") -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=10,
        duration_api_ms=8,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        stop_reason="end_turn",
        total_cost_usd=0.01,
        usage={"input_tokens": 10, "output_tokens": 3},
        result=text,
    )


class _FakeClaudeSDKClient:
    def __init__(self, options, turns):
        self.options = options
        self.turns = deque(turns)
        self.queries: list[tuple[str, str]] = []
        self.connected = False
        self.disconnected = False

    async def connect(self):
        self.connected = True

    async def query(self, prompt, session_id="default"):
        self.queries.append((prompt, session_id))

    async def receive_response(self):
        for message in self.turns.popleft():
            yield message

    async def interrupt(self):
        return None

    async def disconnect(self):
        self.disconnected = True


class _BlockingClaudeSDKClient(_FakeClaudeSDKClient):
    def __init__(self, options):
        super().__init__(options, [])
        self.first_query_started = threading.Event()
        self.release = asyncio.Event()

    async def query(self, prompt, session_id="default"):
        await super().query(prompt, session_id=session_id)
        if len(self.queries) == 1:
            self.first_query_started.set()
        else:
            self.release.set()

    async def receive_response(self):
        await self.release.wait()
        yield _result("steered")

    async def disconnect(self):
        self.release.set()
        await super().disconnect()


class _InterruptibleClaudeSDKClient(_BlockingClaudeSDKClient):
    def __init__(self, options):
        super().__init__(options)
        self.interrupt_called = False

    async def interrupt(self):
        self.interrupt_called = True
        self.release.set()


class _DeniedToolClaudeSDKClient(_FakeClaudeSDKClient):
    async def receive_response(self):
        await self.options.can_use_tool(
            "Bash",
            {"command": "secret-looking command must not leak"},
            MagicMock(),
        )
        yield _result("blocked")


class _QuestionClaudeSDKClient(_FakeClaudeSDKClient):
    def __init__(self, options):
        super().__init__(options, [])
        self.permission = None

    async def receive_response(self):
        self.permission = await self.options.can_use_tool(
            "AskUserQuestion",
            {
                "questions": [
                    {
                        "question": "Which framework?",
                        "header": "Framework",
                        "options": [
                            {"label": "React", "description": "Use React"},
                            {"label": "Vue", "description": "Use Vue"},
                        ],
                        "multiSelect": False,
                        "unknown": "must not leak",
                    }
                ]
            },
            MagicMock(),
        )
        yield _result("continued")


def test_session_routes_ask_user_question_through_input_callback(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    requests = []
    clients = []

    def answer(request):
        requests.append(request)
        return {"question_0": ["React"]}

    def client_factory(options):
        client = _QuestionClaudeSDKClient(options)
        clients.append(client)
        return client

    approval_callback = MagicMock(return_value="deny")
    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        approval_callback=approval_callback,
        input_callback=answer,
        client_factory=client_factory,
    )
    try:
        result = session.run_turn("ask if needed")
    finally:
        session.close()

    assert result.final_text == "continued"
    assert requests == [
        {
            "provider": "claude-code",
            "questions": [
                {
                    "id": "question_0",
                    "header": "Framework",
                    "question": "Which framework?",
                    "options": [
                        {"label": "React", "description": "Use React"},
                        {"label": "Vue", "description": "Use Vue"},
                    ],
                    "multi_select": False,
                    "is_secret": False,
                }
            ],
        }
    ]
    assert "unknown" not in str(requests)
    assert clients[0].permission.behavior == "allow"
    assert clients[0].permission.updated_input["answers"] == {
        "Which framework?": "React"
    }
    approval_callback.assert_not_called()


def test_session_reports_sanitized_approval_denials_per_turn(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    def client_factory(options):
        return _DeniedToolClaudeSDKClient(options, [])

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        approval_callback=MagicMock(return_value="deny"),
        client_factory=client_factory,
    )
    try:
        result = session.run_turn("try a blocked tool")
    finally:
        session.close()

    assert result.approval_denials == [
        {
            "provider": "claude-code",
            "kind": "Bash",
            "reason": "Hermes approval policy denied the request",
        }
    ]
    assert "secret-looking" not in str(result.approval_denials)


def test_session_emits_live_sanitized_tool_progress(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    events = []

    def on_progress(event, name, preview, args, **kwargs):
        events.append((event, name, preview, args, kwargs))

    turns = [
        [
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="tool-1",
                        name="Bash",
                        input={"command": "pwd", "api_key": "secret-looking-value"},
                    )
                ],
                model="claude",
            ),
            UserMessage(
                content=[
                    ToolResultBlock(
                        tool_use_id="tool-1",
                        content="command output",
                        is_error=False,
                    )
                ]
            ),
            _result("done"),
        ]
    ]
    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        tool_progress_callback=on_progress,
        client_factory=lambda options: _FakeClaudeSDKClient(options, turns),
    )
    try:
        session.run_turn("use a tool")
    finally:
        session.close()

    assert [event[0] for event in events] == ["tool.started", "tool.completed"]
    assert [event[1] for event in events] == ["Bash", "Bash"]
    assert events[1][4]["result"] == "command output"
    assert events[1][4]["is_error"] is False
    assert "secret-looking-value" not in str(events)


def test_session_reuses_native_client_across_turns(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    clients = []
    turns = [
        [
            AssistantMessage(
                content=[TextBlock(text="first")],
                model="claude-sonnet-4-6",
            ),
            _result("first"),
        ],
        [
            AssistantMessage(
                content=[TextBlock(text="second")],
                model="claude-sonnet-4-6",
            ),
            _result("second"),
        ],
    ]

    def client_factory(options):
        client = _FakeClaudeSDKClient(options, turns)
        clients.append(client)
        return client

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        system_prompt="Hermes task projection",
        client_factory=client_factory,
    )
    try:
        first = session.run_turn("first request")
        second = session.run_turn("second request")
    finally:
        session.close()

    assert first.final_text == "first"
    assert second.final_text == "second"
    assert first.session_id == second.session_id == "claude-session-1"
    assert len(clients) == 1
    assert clients[0].queries == [
        ("first request", "default"),
        ("second request", "default"),
    ]
    assert clients[0].options.cwd == tmp_path
    assert clients[0].options.system_prompt == {
        "type": "preset",
        "preset": "claude_code",
        "append": "Hermes task projection",
    }
    assert session.resolved_model == "claude-sonnet-4-6"
    assert clients[0].disconnected is True


def test_session_does_not_claim_one_resolved_model_for_a_multi_model_turn(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    client = _FakeClaudeSDKClient(
        MagicMock(),
        [
            [
                AssistantMessage(content=[], model="claude-primary"),
                AssistantMessage(content=[], model="claude-fallback"),
                _result("fallback result"),
            ]
        ],
    )
    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        client_factory=lambda _options: client,
    )
    try:
        session.run_turn("use a fallback if necessary")
    finally:
        session.close()

    assert session.resolved_model is None


def test_session_has_no_hidden_provider_turn_timeout(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(cwd=str(tmp_path), client_factory=MagicMock())

    assert session._turn_timeout is None


def test_session_uses_allowlisted_lazy_sdk_dependency(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(cwd=str(tmp_path), client_factory=MagicMock())

    with patch("tools.lazy_deps.ensure") as ensure:
        session._build_options()

    ensure.assert_called_once_with("delegation.claude-code", prompt=False)


def test_session_resumes_native_claude_history(tmp_path):
    clients = []

    def factory(options):
        client = _FakeClaudeSDKClient(
            options,
            [
                [
                    AssistantMessage(
                        content=[TextBlock(text="resumed")],
                        model="claude-test",
                    ),
                    _result("resumed", session_id="claude-session-42"),
                ]
            ],
        )
        clients.append(client)
        return client

    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(
        cwd=tmp_path,
        resume_session_id="claude-session-42",
        client_factory=factory,
    )
    try:
        result = session.run_turn("continue")
    finally:
        session.close()

    assert result.final_text == "resumed"
    assert clients[0].options.resume == "claude-session-42"


def test_session_maps_hermes_approval_to_claude_permission(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    clients = []
    approvals = []

    def approve(command, description, **kwargs):
        approvals.append((command, description, kwargs))
        return "once"

    def client_factory(options):
        client = _FakeClaudeSDKClient(
            options,
            [[AssistantMessage(content=[TextBlock("done")], model="claude"), _result("done")]],
        )
        clients.append(client)
        return client

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        approval_callback=approve,
        client_factory=client_factory,
    )
    try:
        session.run_turn("use a tool")
        permission = asyncio.run(
            clients[0].options.can_use_tool(
                "Bash",
                {"command": "sleep 1"},
                MagicMock(),
            )
        )
    finally:
        session.close()

    assert permission.behavior == "allow"
    assert approvals[0][0] == "sleep 1"
    assert approvals[0][2]["tool_name"] == "Bash"


def test_session_honors_existing_approval_bypass(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    denied = MagicMock(return_value="deny")
    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        approval_callback=denied,
        approval_bypass=True,
        client_factory=MagicMock(),
    )

    options = session._build_options()
    permission = asyncio.run(
        options.can_use_tool("Bash", {"command": "pwd"}, MagicMock())
    )

    assert permission.behavior == "allow"
    assert permission.updated_input == {"command": "pwd"}
    denied.assert_not_called()


def test_session_steers_an_active_native_turn(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    clients = []
    client_created = threading.Event()

    def client_factory(options):
        client = _BlockingClaudeSDKClient(options)
        clients.append(client)
        client_created.set()
        return client

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        client_factory=client_factory,
    )
    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", session.run_turn("initial request")
        )
    )
    worker.start()
    try:
        assert client_created.wait(timeout=5)
        assert clients[0].first_query_started.wait(timeout=5)
        assert session.request_steer("focus on the parser") is True
        worker.join(timeout=5)
    finally:
        session.close()

    assert worker.is_alive() is False
    assert result_holder["result"].final_text == "steered"
    assert clients[0].queries == [
        ("initial request", "default"),
        ("focus on the parser", "default"),
    ]


def test_session_interrupts_an_active_native_turn(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    clients = []
    client_created = threading.Event()

    def client_factory(options):
        client = _InterruptibleClaudeSDKClient(options)
        clients.append(client)
        client_created.set()
        return client

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        client_factory=client_factory,
    )
    result_holder = {}
    worker = threading.Thread(
        target=lambda: result_holder.setdefault(
            "result", session.run_turn("long request")
        )
    )
    worker.start()
    try:
        assert client_created.wait(timeout=5)
        assert clients[0].first_query_started.wait(timeout=5)
        assert session.request_interrupt() is True
        worker.join(timeout=5)
    finally:
        session.close()

    assert worker.is_alive() is False
    assert clients[0].interrupt_called is True
    assert result_holder["result"].interrupted is True


def test_ai_agent_accepts_claude_agent_sdk_api_mode():
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="http://localhost/claude-agent-sdk",
        api_key="claude-agent-sdk",
        provider="claude-code",
        api_mode="claude_agent_sdk",
        model="claude-code",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        assert agent.api_mode == "claude_agent_sdk"
    finally:
        agent.close()


def test_conversation_loop_hands_turn_to_claude_sdk_runtime():
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="http://localhost/claude-agent-sdk",
        api_key="claude-agent-sdk",
        provider="claude-code",
        api_mode="claude_agent_sdk",
        model="claude-code",
        max_iterations=0,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    native_result = {
        "final_response": "native claude result",
        "messages": [],
        "api_calls": 1,
        "completed": True,
        "interrupted": False,
    }
    agent._run_claude_agent_sdk_turn = MagicMock(return_value=native_result)
    try:
        result = agent.run_conversation("delegate this")
    finally:
        agent.close()

    assert result == native_result
    agent._run_claude_agent_sdk_turn.assert_called_once()


def test_ai_agent_projects_a_native_claude_turn(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeTurnResult
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="http://localhost/claude-agent-sdk",
        api_key="claude-agent-sdk",
        provider="claude-code",
        api_mode="claude_agent_sdk",
        model="claude-code",
        max_iterations=5,
        quiet_mode=True,
        ephemeral_system_prompt="Hermes delegated task",
        skip_context_files=True,
        skip_memory=True,
    )
    agent.session_cwd = str(tmp_path)
    approval_callback = MagicMock(return_value="deny")
    agent._delegate_approval_callback = approval_callback
    agent._native_resume_session_id = "claude-session-42"
    native_session = MagicMock()
    native_session.run_turn.return_value = ClaudeTurnResult(
        final_text="native answer",
        session_id="claude-session-42",
        projected_messages=[{"role": "assistant", "content": "native answer"}],
        usage={"input_tokens": 12, "output_tokens": 4},
        num_turns=2,
        total_cost_usd=0.02,
        approval_denials=[
            {
                "provider": "claude-code",
                "kind": "Edit",
                "reason": "Hermes approval policy denied the request",
            }
        ],
    )

    with (
        patch(
            "agent.claude_agent_sdk_runtime.ClaudeAgentSDKSession",
            return_value=native_session,
        ) as Session,
        patch("tools.approval.is_approval_bypass_active", return_value=True),
    ):
        try:
            result = agent.run_conversation("delegate this")
        finally:
            agent.close()

    assert result["final_response"] == "native answer"
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert result["native_session_id"] == "claude-session-42"
    assert result["approval_denials"][0]["kind"] == "Edit"
    assert result["messages"][-1]["role"] == "assistant"
    assert result["messages"][-1]["content"] == "native answer"
    assert isinstance(result["messages"][-1]["timestamp"], float)
    Session.assert_called_once_with(
        cwd=str(tmp_path),
        system_prompt="Hermes delegated task",
        model=None,
        effort=None,
        max_turns=5,
        permission_mode="default",
        resume_session_id="claude-session-42",
        approval_callback=approval_callback,
        approval_bypass=True,
        tool_progress_callback=getattr(agent, "tool_progress_callback", None),
        input_callback=getattr(agent, "_delegate_input_callback", None),
    )
    native_session.run_turn.assert_called_once_with(
        "delegate this",
        stream_callback=agent._fire_stream_delta,
    )
    assert agent.session_prompt_tokens == 12
    assert agent.session_completion_tokens == 4
    assert agent.session_estimated_cost_usd == 0.02


def test_ai_agent_steer_reaches_active_claude_session():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "claude_agent_sdk"
    agent._claude_sdk_session = MagicMock()
    agent._claude_sdk_session.request_steer.return_value = True

    assert agent.steer("focus on the parser") is True
    agent._claude_sdk_session.request_steer.assert_called_once_with(
        "focus on the parser"
    )


def test_ai_agent_interrupt_reaches_active_claude_session():
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.api_mode = "claude_agent_sdk"
    agent._claude_sdk_session = MagicMock()
    agent._pending_redirect_lock = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._pending_steer_lock = None
    agent.quiet_mode = True

    agent.interrupt("stop", hard_cancel=True)

    agent._claude_sdk_session.request_interrupt.assert_called_once_with()
    assert agent._hard_interrupt_requested.is_set()


def test_ai_agent_close_closes_claude_session():
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="http://localhost/claude-agent-sdk",
        api_key="claude-agent-sdk",
        provider="claude-code",
        api_mode="claude_agent_sdk",
        model="claude-code",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    native_session = MagicMock()
    agent._claude_sdk_session = native_session

    agent.close()

    native_session.close.assert_called_once_with()
    assert agent._claude_sdk_session is None


def test_claude_turn_clears_interrupt_and_preserves_message():
    from agent.claude_agent_sdk_runtime import (
        ClaudeTurnResult,
        run_claude_agent_sdk_turn,
    )

    native_session = MagicMock()
    native_session.run_turn.return_value = ClaudeTurnResult(
        final_text="partial",
        session_id="claude-session-interrupted",
        interrupted=True,
    )
    agent = MagicMock()
    agent._claude_sdk_session = native_session
    agent._interrupt_requested = True
    agent._interrupt_message = "please stop"
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0

    result = run_claude_agent_sdk_turn(
        agent,
        user_message="work",
        original_user_message="work",
        messages=[{"role": "user", "content": "work"}],
        effective_task_id="task-1",
    )

    agent.clear_interrupt.assert_called_once_with()
    assert result["interrupted"] is True
    assert result["interrupt_message"] == "please stop"


def test_claude_turn_retires_failed_sdk_session():
    from agent.claude_agent_sdk_runtime import run_claude_agent_sdk_turn

    native_session = MagicMock()
    native_session.session_id = "claude-session-failed"
    native_session.run_turn.side_effect = RuntimeError("sdk transport died")
    agent = MagicMock()
    agent._claude_sdk_session = native_session
    agent._interrupt_requested = False
    agent._delegate_native_config = {
        "model": "claude-fable-5",
        "effort": "xhigh",
        "approval_mode": "auto",
    }

    result = run_claude_agent_sdk_turn(
        agent,
        user_message="work",
        original_user_message="work",
        messages=[{"role": "user", "content": "work"}],
        effective_task_id="task-1",
    )

    native_session.close.assert_called_once_with()
    assert agent._claude_sdk_session is None
    assert result["completed"] is False
    assert result["error"] == "sdk transport died"
    assert result["native_runtime"] == "claude-code"
    assert result["native_session_id"] == "claude-session-failed"
    assert result["native_model_requested"] == "claude-fable-5"
    assert result["native_effort_requested"] == "xhigh"
    assert result["native_approval_mode_requested"] == "auto"


def test_session_forwards_managed_native_seat_to_sdk_options(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        model="claude-fable-5",
        effort="xhigh",
        permission_mode="auto",
        client_factory=MagicMock(),
    )

    options = session._build_options()

    assert options.model == "claude-fable-5"
    assert options.effort == "xhigh"
    assert options.permission_mode == "auto"


def test_session_auto_permission_mode_keeps_hermes_approval_gate(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    denied = MagicMock(return_value="deny")
    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        permission_mode="auto",
        approval_callback=denied,
        client_factory=MagicMock(),
    )

    options = session._build_options()

    # Classifier auto mode must not shadow can_use_tool: anything the
    # classifier escalates still reaches Hermes' approval policy.
    assert options.permission_mode == "auto"
    assert options.can_use_tool is not None
    permission = asyncio.run(
        options.can_use_tool("Bash", {"command": "rm -rf /"}, MagicMock())
    )
    assert permission.behavior == "deny"
    denied.assert_called_once()


def test_session_fails_fast_on_permission_mode_the_sdk_cannot_honor(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        permission_mode="classifier-please",
        client_factory=MagicMock(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        session._build_options()

    assert "classifier-please" in str(excinfo.value)


def test_session_never_selects_a_permission_mode_that_shadows_approvals(tmp_path):
    from agent.claude_agent_sdk_runtime import ClaudeAgentSDKSession

    session = ClaudeAgentSDKSession(
        cwd=str(tmp_path),
        permission_mode="bypassPermissions",
        approval_callback=MagicMock(return_value="deny"),
        client_factory=MagicMock(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        session._build_options()

    assert "bypassPermissions" in str(excinfo.value)


def test_claude_turn_builds_the_session_from_the_managed_native_seat(tmp_path):
    import agent.claude_agent_sdk_runtime as runtime_module
    from agent.claude_agent_sdk_runtime import (
        ClaudeTurnResult,
        run_claude_agent_sdk_turn,
    )

    built = {}

    def _fake_session(**kwargs):
        built.update(kwargs)
        session = MagicMock()
        session.run_turn.return_value = ClaudeTurnResult(
            final_text="done",
            session_id="claude-session-seat",
            model_usage={"claude-fable-5": {"inputTokens": 10}},
        )
        return session

    agent = MagicMock()
    agent._claude_sdk_session = None
    agent.session_cwd = str(tmp_path)
    agent.max_iterations = 4
    agent._native_resume_session_id = "claude-native-resume-42"
    agent._delegate_native_config = {
        "model": "claude-fable-5",
        "effort": "xhigh",
        "approval_mode": "auto",
    }
    agent._interrupt_requested = False
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent._session_db = None

    with patch.object(runtime_module, "ClaudeAgentSDKSession", _fake_session):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="work",
            original_user_message="work",
            messages=[],
            effective_task_id="task-1",
        )

    assert built["model"] == "claude-fable-5"
    assert built["effort"] == "xhigh"
    assert built["permission_mode"] == "auto"
    assert built["resume_session_id"] == "claude-native-resume-42"
    assert result["native_model_requested"] == "claude-fable-5"
    assert result["native_effort_requested"] == "xhigh"
    assert result["native_approval_mode_requested"] == "auto"
    assert result["native_model_resolved"] == "claude-fable-5"


def test_claude_turn_without_a_managed_seat_keeps_provider_defaults(tmp_path):
    import agent.claude_agent_sdk_runtime as runtime_module
    from agent.claude_agent_sdk_runtime import (
        ClaudeTurnResult,
        run_claude_agent_sdk_turn,
    )

    built = {}

    def _fake_session(**kwargs):
        built.update(kwargs)
        session = MagicMock()
        session.run_turn.return_value = ClaudeTurnResult(
            final_text="done",
            session_id="claude-session-default",
        )
        return session

    agent = MagicMock()
    agent._claude_sdk_session = None
    agent.session_cwd = str(tmp_path)
    agent.max_iterations = 4
    agent._native_resume_session_id = None
    agent._delegate_native_config = None
    agent._interrupt_requested = False
    agent.session_api_calls = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent._session_db = None

    with patch.object(runtime_module, "ClaudeAgentSDKSession", _fake_session):
        result = run_claude_agent_sdk_turn(
            agent,
            user_message="work",
            original_user_message="work",
            messages=[],
            effective_task_id="task-1",
        )

    assert built["model"] is None
    assert built["effort"] is None
    assert built["permission_mode"] == "default"
    assert "native_model_requested" not in result
