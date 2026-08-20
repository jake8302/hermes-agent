#!/usr/bin/env python3
"""Managed native seat configuration for delegate_task.

Covers the model-facing ``native`` object (model / effort / approval_mode),
its provider-aware validation, and the structural plumbing that carries the
request into the Claude Agent SDK and the Codex App Server.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    DELEGATE_TASK_SCHEMA,
    _coerce_native_config,
    _native_classifier_approvals_enabled,
    delegate_task,
)


def _make_mock_parent(depth=0):
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


def test_classifier_approval_gate_reads_active_profile_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert _native_classifier_approvals_enabled() is False

    (tmp_path / "config.yaml").write_text(
        "delegation:\n  native_classifier_approvals: true\n",
        encoding="utf-8",
    )
    assert _native_classifier_approvals_enabled() is True


class TestNativeConfigValidation(unittest.TestCase):
    """Provider-invalid native requests are refused before any child spawns."""

    def test_schema_exposes_native_object(self):
        props = DELEGATE_TASK_SCHEMA["parameters"]["properties"]
        self.assertIn("native", props)
        native_props = props["native"]["properties"]
        self.assertEqual(sorted(native_props), ["approval_mode", "effort", "model"])
        self.assertEqual(
            native_props["approval_mode"]["enum"],
            ["default", "auto", "approve_for_me"],
        )
        task_props = props["tasks"]["items"]["properties"]
        self.assertIn("native", task_props)

    def test_hermes_runtime_rejects_native_options(self):
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    goal="Do the managed work in a Hermes child",
                    runtime="hermes",
                    native={"model": "claude-fable-5"},
                    parent_agent=parent,
                )
            )

        self.assertIn("native", result["error"])
        self.assertIn("claude-code", result["error"])
        MockAgent.assert_not_called()

    def test_approve_for_me_is_rejected_on_claude_code(self):
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    goal="Review the diff with a managed native worker",
                    runtime="claude-code",
                    native={"approval_mode": "approve_for_me"},
                    parent_agent=parent,
                )
            )

        self.assertIn("approve_for_me", result["error"])
        self.assertIn("claude-code", result["error"])
        self.assertIn("'auto'", result["error"])
        MockAgent.assert_not_called()

    def test_auto_is_rejected_on_codex(self):
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    goal="Review the diff with a managed native worker",
                    runtime="codex",
                    native={"approval_mode": "auto"},
                    parent_agent=parent,
                )
            )

        self.assertIn("codex", result["error"])
        self.assertIn("approve_for_me", result["error"])
        MockAgent.assert_not_called()

    def test_effort_is_validated_against_the_provider_contract(self):
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            claude_error = json.loads(
                delegate_task(
                    goal="Run the managed native claude worker on minimal",
                    runtime="claude-code",
                    native={"effort": "minimal"},
                    parent_agent=parent,
                )
            )["error"]

        self.assertIn("low, medium, high, xhigh, max", claude_error)
        MockAgent.assert_not_called()

        for effort in ("max", "ultra"):
            config, error = _coerce_native_config(
                {"effort": effort}, "codex", "delegate_task"
            )
            self.assertIsNone(error)
            self.assertEqual(config, {"effort": effort})

    def test_native_model_rejects_non_strings_and_non_ascii_controls(self):
        for model in (True, 5, "safe\u202eevil"):
            with self.subTest(model=model):
                config, error = _coerce_native_config(
                    {"model": model},
                    "codex",
                    "delegate_task",
                )
                self.assertIsNone(config)
                self.assertIsNotNone(error)

    def test_unknown_native_field_is_rejected(self):
        parent = _make_mock_parent(depth=0)

        with patch("run_agent.AIAgent") as MockAgent:
            result = json.loads(
                delegate_task(
                    goal="Pin a native seat with a bogus knob",
                    runtime="codex",
                    native={"sandbox": "danger-full-access"},
                    parent_agent=parent,
                )
            )

        self.assertIn("sandbox", result["error"])
        MockAgent.assert_not_called()

    def test_classifier_approval_mode_requires_operator_opt_in(self):
        parent = _make_mock_parent(depth=0)
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch("tools.delegate_tool._load_config", return_value={}),
        ):
            result = json.loads(
                delegate_task(
                    goal="Let the provider classifier review approvals",
                    runtime="codex",
                    native={"approval_mode": "approve_for_me"},
                    parent_agent=parent,
                )
            )

        self.assertIn("delegation.native_classifier_approvals", result["error"])
        MockAgent.assert_not_called()


class TestNativeConfigStamping(unittest.TestCase):
    """The validated seat rides on the child, not on prose."""

    def _spawn(self, **kwargs):
        parent = _make_mock_parent(depth=0)
        with (
            patch("run_agent.AIAgent") as MockAgent,
            patch(
                "tools.delegate_tool._resolve_workspace_hint",
                return_value="/tmp/native-seat-child",
            ),
            patch(
                "tools.delegate_tool._native_classifier_approvals_enabled",
                return_value=True,
            ),
        ):
            mock_child = MagicMock()
            mock_child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }
            MockAgent.return_value = mock_child
            payload = json.loads(delegate_task(parent_agent=parent, **kwargs))
        return payload, mock_child

    def test_claude_seat_is_stamped_on_the_child(self):
        payload, child = self._spawn(
            goal="Do the managed native Claude work",
            runtime="claude-code",
            native={
                "model": "claude-fable-5",
                "effort": "xhigh",
                "approval_mode": "auto",
            },
        )

        self.assertEqual(payload["results"][0]["summary"], "ok")
        self.assertEqual(
            child._delegate_native_config,
            {
                "model": "claude-fable-5",
                "effort": "xhigh",
                "approval_mode": "auto",
            },
        )

    def test_codex_seat_is_stamped_on_the_child(self):
        _payload, child = self._spawn(
            goal="Do the managed native Codex work",
            runtime="codex",
            native={
                "model": "gpt-5.1-codex-max",
                "effort": "xhigh",
                "approval_mode": "approve_for_me",
            },
        )

        self.assertEqual(
            child._delegate_native_config,
            {
                "model": "gpt-5.1-codex-max",
                "effort": "xhigh",
                "approval_mode": "approve_for_me",
            },
        )

    def test_omitted_native_leaves_no_seat_override(self):
        _payload, child = self._spawn(
            goal="Do the managed native work with provider defaults",
            runtime="claude-code",
        )

        self.assertIsNone(child._delegate_native_config)

    def test_per_task_native_replaces_top_level_native_without_merging(self):
        parent = _make_mock_parent(depth=0)
        children = [MagicMock(), MagicMock()]
        for child in children:
            child.run_conversation.return_value = {
                "final_response": "ok",
                "completed": True,
                "api_calls": 1,
            }

        with (
            patch("run_agent.AIAgent", side_effect=children),
            patch(
                "tools.delegate_tool._resolve_workspace_hint",
                return_value="/tmp/native-seat-batch",
            ),
        ):
            payload = json.loads(
                delegate_task(
                    tasks=[
                        {
                            "goal": "Use the task-specific Claude seat",
                            "runtime": "claude-code",
                            "native": {"model": "task-claude-model"},
                        },
                        {
                            "goal": "Use the inherited Codex seat",
                            "runtime": "codex",
                        },
                    ],
                    runtime="codex",
                    native={"model": "top-codex-model", "effort": "xhigh"},
                    parent_agent=parent,
                    background=False,
                )
            )

        self.assertNotIn("error", payload)
        self.assertEqual(
            children[0]._delegate_native_config,
            {"model": "task-claude-model"},
        )
        self.assertEqual(
            children[1]._delegate_native_config,
            {"model": "top-codex-model", "effort": "xhigh"},
        )


if __name__ == "__main__":
    unittest.main()
