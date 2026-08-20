"""Persistent Claude Agent SDK session for provider-native Hermes turns."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)

ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]


_SECRET_PROGRESS_KEYS = frozenset(
    {"api_key", "apikey", "authorization", "cookie", "password", "secret", "token"}
)


def _sanitize_progress_value(value: Any, *, key: str = "") -> Any:
    normalized_key = key.strip().lower().replace("-", "_")
    if normalized_key in _SECRET_PROGRESS_KEYS or normalized_key.endswith(
        ("_key", "_password", "_secret", "_token")
    ):
        return "«redacted-secret»"
    if isinstance(value, dict):
        return {
            str(item_key)[:80]: _sanitize_progress_value(item_value, key=str(item_key))
            for item_key, item_value in list(value.items())[:50]
        }
    if isinstance(value, list):
        return [_sanitize_progress_value(item) for item in value[:50]]
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)[:4000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:4000]


@dataclass
class ClaudeTurnResult:
    final_text: str = ""
    session_id: str = ""
    projected_messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    num_turns: int = 0
    total_cost_usd: Optional[float] = None
    interrupted: bool = False
    error: str = ""
    approval_denials: list[dict[str, str]] = field(default_factory=list)
    model_usage: dict[str, Any] = field(default_factory=dict)


class ClaudeAgentSDKSession:
    """Own one long-lived SDK client and its event loop thread."""

    def __init__(
        self,
        *,
        cwd: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[ClaudeEffort] = None,
        max_turns: Optional[int] = None,
        permission_mode: str = "default",
        resume_session_id: Optional[str] = None,
        approval_callback: Optional[Callable[..., str]] = None,
        approval_bypass: bool = False,
        tool_progress_callback: Optional[Callable[..., None]] = None,
        input_callback: Optional[
            Callable[[dict[str, Any]], Optional[dict[str, list[str]]]]
        ] = None,
        client_factory: Optional[Callable[[Any], Any]] = None,
        turn_timeout: Optional[float] = None,
    ) -> None:
        self._cwd = str(cwd)
        self._system_prompt = system_prompt
        self._model = model
        self._effort = effort
        self._max_turns = max_turns
        self._permission_mode = permission_mode
        self._resume_session_id = resume_session_id
        self._approval_callback = approval_callback
        self._approval_bypass = bool(approval_bypass)
        self._tool_progress_callback = tool_progress_callback
        self._input_callback = input_callback
        self._client_factory = client_factory
        self._turn_timeout = turn_timeout

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._client: Any = None
        self._closed = False
        self._turn_lock = threading.Lock()
        self._active = threading.Event()
        self._interrupt_requested = threading.Event()
        self._session_id = ""
        self._resolved_model: Optional[str] = None
        self._turn_approval_denials: list[dict[str, str]] = []
        self._tool_starts: dict[str, tuple[str, float]] = {}

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def resolved_model(self) -> Optional[str]:
        return self._resolved_model

    def _start_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        if self._closed:
            raise RuntimeError("Claude Agent SDK session is closed")

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            loop.run_forever()
            loop.close()

        self._loop_thread = threading.Thread(
            target=_run_loop,
            name="hermes-claude-agent-sdk",
            daemon=True,
        )
        self._loop_thread.start()
        if not self._loop_ready.wait(timeout=10):
            raise TimeoutError("Claude Agent SDK event loop did not start")
        assert self._loop is not None
        return self._loop

    def _submit(self, coroutine, *, timeout: Optional[float] = None):
        loop = self._start_loop()
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        return future.result(timeout=timeout)

    def _build_options(self):
        from tools.lazy_deps import FeatureUnavailable, ensure

        try:
            ensure("delegation.claude-code", prompt=False)
        except FeatureUnavailable as exc:
            raise RuntimeError(
                "The claude-code delegation runtime requires the optional "
                f"claude-agent-sdk package: {exc}"
            ) from exc

        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                PermissionResultAllow,
                PermissionResultDeny,
            )
        except ImportError as exc:  # pragma: no cover - exercised by integration preflight
            raise RuntimeError(
                "The claude-code delegation runtime requires the optional "
                "claude-agent-sdk package. Install Hermes with the "
                "claude-agent-sdk extra."
            ) from exc

        if self._permission_mode not in {"default", "auto"}:
            raise RuntimeError(
                "Managed Claude Code delegation supports permission_mode "
                f"'default' or 'auto', not {self._permission_mode!r}. "
                "Modes that bypass or suppress Hermes approval handling are "
                "not allowed."
            )

        appended_prompt: Any = None
        if self._system_prompt:
            appended_prompt = {
                "type": "preset",
                "preset": "claude_code",
                "append": self._system_prompt,
            }

        can_use_tool = None
        if (
            self._approval_bypass
            or self._approval_callback is not None
            or self._input_callback is not None
        ):
            approval_callback = self._approval_callback
            input_callback = self._input_callback

            async def _can_use_tool(tool_name, tool_input, context):
                if tool_name == "AskUserQuestion" and input_callback is not None:
                    questions = []
                    question_text_by_id = {}
                    for index, raw_question in enumerate(
                        (tool_input.get("questions") or [])[:8]
                    ):
                        if not isinstance(raw_question, dict):
                            continue
                        question_text = str(
                            raw_question.get("question") or ""
                        )[:2000]
                        if not question_text:
                            continue
                        question_id = f"question_{index}"
                        question_text_by_id[question_id] = question_text
                        options = []
                        for raw_option in (raw_question.get("options") or [])[:8]:
                            if not isinstance(raw_option, dict):
                                continue
                            options.append(
                                {
                                    "label": str(
                                        raw_option.get("label") or ""
                                    )[:200],
                                    "description": str(
                                        raw_option.get("description") or ""
                                    )[:500],
                                }
                            )
                        questions.append(
                            {
                                "id": question_id,
                                "header": str(
                                    raw_question.get("header") or ""
                                )[:200],
                                "question": question_text,
                                "options": options,
                                "multi_select": bool(
                                    raw_question.get("multiSelect", False)
                                ),
                                "is_secret": bool(
                                    raw_question.get("isSecret", False)
                                ),
                            }
                        )
                    try:
                        raw_answers = await asyncio.to_thread(
                            input_callback,
                            {"provider": "claude-code", "questions": questions},
                        )
                    except Exception:
                        logger.exception(
                            "input_callback raised on Claude AskUserQuestion"
                        )
                        raw_answers = None
                    if not isinstance(raw_answers, dict):
                        return PermissionResultDeny(
                            message="Hermes could not collect an answer."
                        )
                    answers = {}
                    for question_id, values in raw_answers.items():
                        question_id = str(question_id)
                        question_text = question_text_by_id.get(question_id)
                        if question_text is None or not isinstance(values, list):
                            continue
                        answers[question_text] = ", ".join(
                            str(value)[:2000] for value in values[:8]
                        )
                    return PermissionResultAllow(
                        updated_input={**tool_input, "answers": answers}
                    )
                if self._approval_bypass:
                    return PermissionResultAllow(updated_input=tool_input)
                command = str(tool_input.get("command") or tool_name)
                assert approval_callback is not None
                decision = approval_callback(
                    command,
                    f"Claude Code requests {tool_name}",
                    tool_name=tool_name,
                    tool_input=tool_input,
                    permission_context=context,
                )
                if str(decision or "").strip().lower() in {
                    "allow",
                    "always",
                    "once",
                    "yes",
                }:
                    return PermissionResultAllow(updated_input=tool_input)
                self._turn_approval_denials.append(
                    {
                        "provider": "claude-code",
                        "kind": str(tool_name or "tool"),
                        "reason": "Hermes approval policy denied the request",
                    }
                )
                return PermissionResultDeny(
                    message="Hermes delegation policy denied this tool request."
                )

            can_use_tool = _can_use_tool

        return ClaudeAgentOptions(
            cwd=Path(self._cwd),
            cli_path=shutil.which("claude"),
            model=self._model,
            effort=self._effort,
            max_turns=self._max_turns,
            permission_mode=self._permission_mode,
            system_prompt=appended_prompt,
            setting_sources=["user", "project", "local"],
            resume=self._resume_session_id,
            include_partial_messages=True,
            can_use_tool=can_use_tool,
        )

    async def _ensure_connected(self):
        if self._client is not None:
            return self._client
        options = self._build_options()
        if self._client_factory is None:
            from claude_agent_sdk import ClaudeSDKClient

            factory = ClaudeSDKClient
        else:
            factory = self._client_factory
        client = factory(options)
        await client.connect()
        self._client = client
        return client

    @staticmethod
    def _stream_text(message: Any) -> str:
        if type(message).__name__ != "StreamEvent":
            return ""
        event = getattr(message, "event", None)
        if not isinstance(event, dict):
            return ""
        delta = event.get("delta")
        if not isinstance(delta, dict) or delta.get("type") != "text_delta":
            return ""
        text = delta.get("text")
        return text if isinstance(text, str) else ""

    @staticmethod
    def _project_assistant(message: Any) -> tuple[str, Optional[dict[str, Any]]]:
        if type(message).__name__ != "AssistantMessage":
            return "", None
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in getattr(message, "content", []) or []:
            block_type = type(block).__name__
            if block_type == "TextBlock":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
            elif block_type == "ToolUseBlock":
                tool_calls.append(
                    {
                        "id": str(getattr(block, "id", "") or ""),
                        "type": "function",
                        "function": {
                            "name": str(getattr(block, "name", "") or ""),
                            "arguments": json.dumps(
                                getattr(block, "input", {}) or {},
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
        text = "\n".join(text_parts)
        projected: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            projected["tool_calls"] = tool_calls
        if not text and not tool_calls:
            return "", None
        return text, projected

    @staticmethod
    def _project_tool_results(message: Any) -> list[dict[str, Any]]:
        if type(message).__name__ != "UserMessage":
            return []
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return []
        projected: list[dict[str, Any]] = []
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            raw_content = getattr(block, "content", "")
            if isinstance(raw_content, str):
                rendered = raw_content
            else:
                rendered = json.dumps(raw_content, ensure_ascii=False)
            projected.append(
                {
                    "role": "tool",
                    "tool_call_id": str(
                        getattr(block, "tool_use_id", "") or ""
                    ),
                    "content": rendered,
                }
            )
        return projected

    def _emit_tool_progress(self, message: Any) -> None:
        callback = self._tool_progress_callback
        if callback is None:
            return
        try:
            if type(message).__name__ == "AssistantMessage":
                from agent.tool_executor import _build_tool_preview

                for block in getattr(message, "content", []) or []:
                    if type(block).__name__ != "ToolUseBlock":
                        continue
                    tool_name = str(getattr(block, "name", "") or "tool")
                    tool_input = getattr(block, "input", {}) or {}
                    display_args = _sanitize_progress_value(tool_input)
                    if not isinstance(display_args, dict):
                        display_args = {}
                    tool_id = str(getattr(block, "id", "") or "")
                    if tool_id:
                        self._tool_starts[tool_id] = (tool_name, time.monotonic())
                    callback(
                        "tool.started",
                        tool_name,
                        _build_tool_preview(tool_name, display_args),
                        display_args,
                    )
                return

            if type(message).__name__ != "UserMessage":
                return
            from agent.redact import redact_terminal_output

            for block in getattr(message, "content", []) or []:
                if type(block).__name__ != "ToolResultBlock":
                    continue
                tool_id = str(getattr(block, "tool_use_id", "") or "")
                started = self._tool_starts.pop(tool_id, None)
                tool_name = started[0] if started is not None else "claude_tool"
                duration = (
                    max(0.0, time.monotonic() - started[1])
                    if started is not None
                    else 0.0
                )
                raw_content = getattr(block, "content", "")
                rendered = (
                    raw_content
                    if isinstance(raw_content, str)
                    else json.dumps(raw_content, ensure_ascii=False)
                )
                callback(
                    "tool.completed",
                    tool_name,
                    None,
                    None,
                    duration=duration,
                    is_error=bool(getattr(block, "is_error", False)),
                    result=redact_terminal_output(rendered, force=True)[:4000],
                )
        except Exception:
            logger.debug("Claude SDK tool progress callback raised", exc_info=True)

    async def _run_turn_async(
        self,
        prompt: str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> ClaudeTurnResult:
        self._turn_approval_denials = []
        client = await self._ensure_connected()
        await client.query(prompt, session_id="default")

        result = ClaudeTurnResult()
        assistant_text: list[str] = []
        observed_assistant_models: set[str] = set()
        self._resolved_model = None
        async for message in client.receive_response():
            if type(message).__name__ == "AssistantMessage":
                assistant_model = getattr(message, "model", None)
                if isinstance(assistant_model, str) and assistant_model:
                    observed_assistant_models.add(assistant_model)
            delta = self._stream_text(message)
            if delta and stream_callback is not None:
                stream_callback(delta)

            self._emit_tool_progress(message)

            text, projected_assistant = self._project_assistant(message)
            if text:
                assistant_text.append(text)
            if projected_assistant is not None:
                result.projected_messages.append(projected_assistant)
            result.projected_messages.extend(self._project_tool_results(message))

            if type(message).__name__ == "ResultMessage":
                result.session_id = str(
                    getattr(message, "session_id", "") or ""
                )
                self._session_id = result.session_id
                result.num_turns = int(getattr(message, "num_turns", 0) or 0)
                usage = getattr(message, "usage", None)
                result.usage = dict(usage) if isinstance(usage, dict) else {}
                model_usage = getattr(message, "model_usage", None)
                result.model_usage = (
                    dict(model_usage) if isinstance(model_usage, dict) else {}
                )
                total_cost = getattr(message, "total_cost_usd", None)
                result.total_cost_usd = (
                    float(total_cost) if total_cost is not None else None
                )
                message_result = getattr(message, "result", None)
                if isinstance(message_result, str):
                    result.final_text = message_result
                if bool(getattr(message, "is_error", False)):
                    errors = getattr(message, "errors", None)
                    result.error = (
                        "; ".join(str(item) for item in errors)
                        if isinstance(errors, list) and errors
                        else str(getattr(message, "subtype", "Claude SDK error"))
                    )

        observed_models = observed_assistant_models or {
            model
            for model in result.model_usage
            if isinstance(model, str) and model
        }
        if len(observed_models) == 1:
            self._resolved_model = next(iter(observed_models))
        if not result.final_text:
            result.final_text = "\n".join(assistant_text).strip()
        result.approval_denials = list(self._turn_approval_denials)
        return result

    def run_turn(
        self,
        prompt: str,
        *,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> ClaudeTurnResult:
        if self._closed:
            raise RuntimeError("Claude Agent SDK session is closed")
        with self._turn_lock:
            self._interrupt_requested.clear()
            self._active.set()
            try:
                result = self._submit(
                    self._run_turn_async(prompt, stream_callback),
                    timeout=self._turn_timeout,
                )
                result.interrupted = self._interrupt_requested.is_set()
                return result
            finally:
                self._active.clear()

    async def _steer(self, text: str) -> bool:
        client = self._client
        if client is None:
            return False
        await client.query(text, session_id="default")
        return True

    def request_steer(self, text: str) -> bool:
        """Queue user guidance into the currently active SDK turn."""
        cleaned = str(text or "").strip()
        if not cleaned or not self._active.is_set() or self._client is None:
            return False
        try:
            return bool(self._submit(self._steer(cleaned), timeout=10))
        except Exception:
            logger.debug("Claude Agent SDK steer failed", exc_info=True)
            return False

    async def _interrupt(self) -> bool:
        client = self._client
        if client is None:
            return False
        await client.interrupt()
        return True

    def request_interrupt(self) -> bool:
        """Interrupt the active turn without deleting the native session."""
        if not self._active.is_set() or self._client is None:
            return False
        self._interrupt_requested.set()
        try:
            return bool(self._submit(self._interrupt(), timeout=10))
        except Exception:
            logger.debug("Claude Agent SDK interrupt failed", exc_info=True)
            return False

    async def _disconnect(self) -> None:
        if self._client is None:
            return
        client, self._client = self._client, None
        await client.disconnect()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        if loop is None:
            return
        try:
            self._submit(self._disconnect(), timeout=15)
        except Exception:
            logger.debug("Claude Agent SDK disconnect failed", exc_info=True)
        loop.call_soon_threadsafe(loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=15)
        self._loop = None


def run_claude_agent_sdk_turn(
    agent: Any,
    *,
    user_message: str,
    original_user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> dict[str, Any]:
    """Run one Hermes turn through a persistent native Claude session."""
    del original_user_message, effective_task_id, should_review_memory

    session = getattr(agent, "_claude_sdk_session", None)
    native_config = getattr(agent, "_delegate_native_config", None)
    if not isinstance(native_config, dict):
        native_config = {}
    requested_native_metadata = {
        key: value
        for key, value in {
            "native_model_requested": native_config.get("model"),
            "native_effort_requested": native_config.get("effort"),
            "native_approval_mode_requested": native_config.get("approval_mode"),
        }.items()
        if isinstance(value, str) and value
    }
    if session is None:
        from tools.approval import is_approval_bypass_active

        session = ClaudeAgentSDKSession(
            cwd=str(getattr(agent, "session_cwd", None) or Path.cwd()),
            system_prompt=getattr(agent, "ephemeral_system_prompt", None),
            model=native_config.get("model"),
            effort=native_config.get("effort"),
            max_turns=getattr(agent, "max_iterations", None),
            permission_mode=(
                "auto"
                if native_config.get("approval_mode") == "auto"
                else "default"
            ),
            resume_session_id=getattr(agent, "_native_resume_session_id", None),
            approval_callback=getattr(agent, "_delegate_approval_callback", None),
            approval_bypass=is_approval_bypass_active(),
            tool_progress_callback=getattr(agent, "tool_progress_callback", None),
            input_callback=getattr(agent, "_delegate_input_callback", None),
        )
        agent._claude_sdk_session = session

    try:
        native_result = session.run_turn(
            str(user_message),
            stream_callback=getattr(agent, "_fire_stream_delta", None),
        )
    except Exception as exc:
        logger.exception("Claude Agent SDK turn failed")
        native_session_id = (
            getattr(session, "session_id", None)
            or getattr(agent, "_native_resume_session_id", None)
        )
        try:
            session.close()
        except Exception:
            pass
        agent._claude_sdk_session = None
        user_interrupted = bool(getattr(agent, "_interrupt_requested", False))
        interrupt_message = (
            getattr(agent, "_interrupt_message", None)
            if user_interrupted
            else None
        )
        if user_interrupted:
            agent.clear_interrupt()
        return {
            "final_response": f"Claude Agent SDK turn failed: {exc}",
            "messages": messages,
            "api_calls": 0,
            "completed": False,
            "partial": True,
            "interrupted": user_interrupted,
            "native_runtime": "claude-code",
            "native_session_id": native_session_id,
            **requested_native_metadata,
            **(
                {"interrupt_message": interrupt_message}
                if interrupt_message
                else {}
            ),
            "error": str(exc),
        }

    user_interrupted = bool(
        native_result.interrupted
        and getattr(agent, "_interrupt_requested", False)
    )
    interrupt_message = (
        getattr(agent, "_interrupt_message", None) if user_interrupted else None
    )
    if user_interrupted:
        agent.clear_interrupt()
    if native_result.projected_messages:
        from agent.message_metadata import append_message

        for projected_message in native_result.projected_messages:
            append_message(messages, projected_message)

        if getattr(agent, "_session_db", None) is not None:
            try:
                flush_ok = agent._flush_messages_to_session_db(messages)
            except Exception:
                flush_ok = False
                logger.warning(
                    "Claude Agent SDK projected-message flush failed",
                    exc_info=True,
                )
            if flush_ok is False:
                logger.warning(
                    "Claude Agent SDK turn was delivered but could not be "
                    "persisted to SessionDB"
                )

    api_calls = max(1, int(native_result.num_turns or 0))
    agent.session_api_calls = int(getattr(agent, "session_api_calls", 0)) + api_calls
    input_tokens = int(native_result.usage.get("input_tokens", 0) or 0)
    output_tokens = int(native_result.usage.get("output_tokens", 0) or 0)
    agent.session_prompt_tokens = int(
        getattr(agent, "session_prompt_tokens", 0)
    ) + input_tokens
    agent.session_completion_tokens = int(
        getattr(agent, "session_completion_tokens", 0)
    ) + output_tokens
    agent.session_total_tokens = int(getattr(agent, "session_total_tokens", 0)) + (
        input_tokens + output_tokens
    )
    agent.session_estimated_cost_usd = float(
        getattr(agent, "session_estimated_cost_usd", 0.0)
    ) + float(native_result.total_cost_usd or 0.0)
    agent._last_turn_usage = dict(native_result.usage)
    agent._claude_native_session_id = native_result.session_id

    observed_models = [
        model
        for model in native_result.model_usage
        if isinstance(model, str) and model
    ]
    resolved_native_metadata = (
        {"native_model_resolved": observed_models[0]}
        if len(observed_models) == 1
        else {}
    )

    completed = not native_result.error and not native_result.interrupted
    result = {
        "final_response": native_result.final_text,
        "messages": messages,
        "api_calls": api_calls,
        "completed": completed,
        "interrupted": native_result.interrupted,
        "native_runtime": "claude-code",
        "native_session_id": native_result.session_id,
        "approval_denials": list(native_result.approval_denials),
        "agent_persisted": True,
        **requested_native_metadata,
        **resolved_native_metadata,
        **(
            {"interrupt_message": interrupt_message}
            if interrupt_message
            else {}
        ),
    }
    if native_result.error:
        result["error"] = native_result.error
    return result
