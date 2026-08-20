---
name: codex
description: "Route managed Codex subagents; run standalone CLI sessions."
version: 1.2.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Coding-Agent, Codex, OpenAI, Code-Review, Refactoring]
    related_skills: [claude-code, hermes-agent]
---

# Codex CLI

Use [Codex](https://github.com/openai/codex) either as a Hermes-managed native
subagent or as a standalone CLI. Codex is OpenAI's autonomous coding agent.

## Routing precedence

When the user asks to **delegate to**, **spawn**, or **use a native Codex
subagent**, call:

```
delegate_task(runtime="codex", goal="<self-contained task>", context="<needed context>")
```

When the user specifies a native Codex model, effort, or approve-for-me mode,
pass it structurally rather than mentioning it only in the goal:

```
delegate_task(
    runtime="codex",
    native={"model": requested_model, "effort": requested_effort, "approval_mode": "approve_for_me"},
    goal="<self-contained task>",
    context="<needed context>",
)
```

`approval_mode="approve_for_me"` maps to Codex App Server's classifier-backed
reviewer (`approvalPolicy="on-request"` plus
`approvalsReviewer="auto_review"`). It never means `approvalPolicy="never"`
and does not disable the sandbox. Model, effort, and approval mode are
independent. Because the provider classifier may resolve requests before
Hermes' callback, the operator must first opt in with
`hermes config set delegation.native_classifier_approvals true`. If that gate
is off, omit `approval_mode` or surface the pre-dispatch error; never bypass it.
Omitted fields keep provider defaults. Trust the completion's
`native_*_resolved` metadata—not the child's prose—when verifying the seat.
Codex App Server does not echo a per-turn effort override, so effort remains
`native_effort_requested` rather than a resolved claim.
Never fall back to `codex exec` or `codex review` when managed configuration
fails; surface the error instead.

This keeps Hermes responsible for worker identity, background completion,
steering, stopping, resume, and cleanup while Codex App Server owns the native
thread. Do **not** replace this managed path with `codex exec`, `codex review`,
or another `terminal` call.

If the managed child reports `waiting_for_input`, ask the human with `clarify`,
then call `delegate_task(runtime="codex", action="respond", ...)` with the exact
subagent ID, request ID, and question IDs from the request. Never invent the
answer. Stop the child if the human declines. Secret-input requests fail closed
and must not be relayed through model-visible text.

Use the standalone CLI instructions below only when the user explicitly asks to
run the Codex CLI, wants a visible interactive terminal, or needs a CLI-only
command.

## When to use the standalone CLI

- Explicit standalone or visible Codex CLI sessions
- Building features
- Refactoring
- PR reviews
- Batch issue fixing

Requires the codex CLI and a git repository.

## Prerequisites

- Codex installed: `npm install -g @openai/codex`
- OpenAI auth configured: either `OPENAI_API_KEY` or Codex OAuth credentials
  from the Codex CLI login flow
- **Must run inside a git repository** — Codex refuses to run outside one
- Use `pty=true` in terminal calls — Codex is an interactive terminal app

For Hermes itself, `model.provider: openai-codex` uses Hermes-managed Codex
OAuth from `~/.hermes/auth.json` after `hermes auth add openai-codex`. For the
standalone Codex CLI, a valid CLI OAuth session may live under
`~/.codex/auth.json`; do not treat a missing `OPENAI_API_KEY` alone as proof
that Codex auth is missing.

## One-Shot Tasks

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

For scratch work (Codex needs a git repo):
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## Background Mode (Long Tasks)

```
# Start in background with PTY
terminal(command="codex exec --sandbox workspace-write 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Codex asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Key Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | One-shot execution, exits when done |
| `--sandbox workspace-write` (`-s`) | Sandboxed but auto-approves file changes in the workspace (the recommended auto-build mode) |
| `--dangerously-bypass-approvals-and-sandbox` | No sandbox, no approvals (fastest, most dangerous; `--yolo` still works as a hidden alias) |
| `--sandbox danger-full-access` | No Codex sandbox; useful when the host service context breaks bubblewrap |

> **Deprecated:** `--full-auto` still works but the live CLI warns to use `--sandbox workspace-write` instead.

## Hermes Gateway Caveat

When invoking the Codex CLI from a Hermes gateway/service context (for example,
Telegram-driven agent sessions), Codex `workspace-write` sandboxing may fail even
when the same command works in the user's interactive shell. A typical symptom is
bubblewrap/user-namespace errors such as `setting up uid map: Permission denied`
or `loopback: Failed RTM_NEWADDR: Operation not permitted`.

In that context, prefer:

```
codex exec --sandbox danger-full-access "<task>"
```

Use process boundaries as the safety layer instead: explicit `workdir`, clean git
status before launch, narrow task prompts, `git diff` review, targeted tests, and
human/agent confirmation before committing broad changes.

## PR Reviews

Clone to a temp directory for safe review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## Parallel Issue Fixing with Worktrees

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Codex in each
terminal(command="codex --sandbox workspace-write exec 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex --sandbox workspace-write exec 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## Batch PR Reviews

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## Rules for standalone Codex CLI sessions

1. **For standalone Codex CLI runs, always use `pty=true`** — Codex is an interactive terminal app and hangs without a PTY
2. **Git repo required** — Codex won't run outside a git directory. Use `mktemp -d && git init` for scratch
3. **For standalone one-shots, use `exec`** — `codex exec "prompt"` runs and exits cleanly
4. **`--sandbox workspace-write` for building** — auto-approves changes within the sandbox (`--full-auto` is deprecated for this)
5. **Background for long tasks** — use `background=true` and monitor with `process` tool
6. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
7. **Parallel is fine** — run multiple Codex processes at once for batch work

## Routing reminder

For a managed Codex subagent, always call
`delegate_task(runtime="codex", goal="...")`. The standalone rules above apply
only when the user explicitly asks for the Codex CLI or a visible interactive
terminal session.
