from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv
from openai import OpenAI

COMMAND_BLOCK_BEGIN = "==========BELOW ARE COMMAND=========="
COMMAND_BLOCK_END = "==========ABOVE ARE COMMAND=========="

MODEL_CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "x-ai/grok-3": 131072,
    "x-ai/grok-3-mini": 131072,
    "x-ai/grok-4": 256000,
    "x-ai/grok-4-fast": 256000,
    "x-ai/grok-4.1-fast": 256000,
    "x-ai/grok-code-fast-1": 131072,
    "openai/gpt-4o": 128000,
    "openai/gpt-4o-mini": 128000,
    "openai/gpt-5.2": 400000,
    "google/gemini-2.0-flash-001": 1048576,
    "anthropic/claude-3.5-sonnet": 200000,
}

MODEL_CONTEXT_WINDOW_PREFIXES: list[tuple[str, int]] = [
    ("x-ai/grok-", 131072),
    ("openai/gpt-4o", 128000),
    ("openai/gpt-5", 400000),
    ("google/gemini-", 1048576),
    ("anthropic/claude-", 200000),
]

DEFAULT_SYSTEM_PROMPT_RELATIVE_PATH = Path("templates/prompts/system.md")
DEFAULT_GUARDRAILS_PROMPT_RELATIVE_PATH = Path("templates/prompts/runtime_guardrails.md")
DEFAULT_AGENTS_PROFILE = "project_engineer"
AGENTS_PROFILE_TEMPLATE_MAP: dict[str, Path] = {
    "project_engineer": Path("templates/agents/project_engineer.AGENTS.md"),
    "experimental_creature": Path("templates/agents/experimental_creature.AGENTS.md"),
}

DEFAULT_NO_COMMAND_PROMPT = (
    "No command block was detected. Continue by returning one or more commands using:\n"
    "==========BELOW ARE COMMAND==========\n"
    "<command>\n"
    "==========ABOVE ARE COMMAND=========="
)

DEFAULT_PING_PROMPT = (
    "Ping: continue autonomous progress in this project. Do not ask questions. "
    "Choose the next concrete action and execute it if needed."
)

DEFAULT_CONFIG_TOML = """[openrouter]
model = \"x-ai/grok-3\"
base_url = \"https://openrouter.ai/api/v1\"
api_key_env = \"OPENROUTER_API_KEY\"
context_window_tokens = 0

[agent]
max_steps = 0
loop_sleep_seconds = 1.0
max_context_messages = 20
max_message_chars = 4000

[execution]
shell = \"/bin/bash\"
timeout_seconds = 120
max_output_chars = 8000
"""

DEFAULT_MEMORY_MD = """# aibot memory

This file is owned and managed by aibot.
"""


@dataclass(slots=True)
class OpenRouterConfig:
    model: str
    base_url: str
    api_key_env: str
    context_window_tokens: int


@dataclass(slots=True)
class AgentConfig:
    max_steps: int
    loop_sleep_seconds: float
    max_context_messages: int
    max_message_chars: int


@dataclass(slots=True)
class ExecutionConfig:
    shell: str
    timeout_seconds: int
    max_output_chars: int


@dataclass(slots=True)
class AppConfig:
    openrouter: OpenRouterConfig
    agent: AgentConfig
    execution: ExecutionConfig


@dataclass(slots=True)
class CommandResult:
    output: str
    exit_code: int | None
    timed_out: bool


@dataclass(slots=True)
class ContextWindowState:
    model: str
    source: str
    configured_window_tokens: int
    last_prompt_tokens: int
    last_completion_tokens: int
    last_total_tokens: int
    rolling_prompt_tokens: int
    rolling_completion_tokens: int
    rolling_total_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="aibot CLI for project-scoped autonomous runs")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize aibot files in a project directory")
    init_parser.add_argument(
        "--project-dir",
        default=".",
        help="Target project directory (default: current directory).",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing aibot files.",
    )
    init_parser.add_argument(
        "--agents-profile",
        choices=sorted(AGENTS_PROFILE_TEMPLATE_MAP.keys()),
        default=DEFAULT_AGENTS_PROFILE,
        help="AGENTS.md profile to initialize in project directory.",
    )

    run_parser = subparsers.add_parser("run", help="Run aibot loop in a project directory")
    run_parser.add_argument(
        "initial_prompt",
        nargs="?",
        default=None,
        help="Initial request. If omitted, reads stdin; otherwise uses ping.",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="Optional explicit config path (relative to project dir or absolute).",
    )
    run_parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to operate in (default: current directory).",
    )

    parser.add_argument(
        "initial_prompt",
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--project-dir", default=".", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.command is None:
        args.command = "run"
    return args


def truncate_text(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n...[truncated]"


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    return build_config(raw)


def load_default_config() -> AppConfig:
    raw = tomllib.loads(DEFAULT_CONFIG_TOML)
    return build_config(raw)


def build_config(raw: dict[str, Any]) -> AppConfig:

    openrouter_raw = raw.get("openrouter", {})
    agent_raw = raw.get("agent", {})
    execution_raw = raw.get("execution", {})

    return AppConfig(
        openrouter=OpenRouterConfig(
            model=str(openrouter_raw.get("model", "x-ai/grok-3")),
            base_url=str(openrouter_raw.get("base_url", "https://openrouter.ai/api/v1")),
            api_key_env=str(openrouter_raw.get("api_key_env", "OPENROUTER_API_KEY")),
            context_window_tokens=int(openrouter_raw.get("context_window_tokens", 0)),
        ),
        agent=AgentConfig(
            max_steps=int(agent_raw.get("max_steps", 0)),
            loop_sleep_seconds=float(agent_raw.get("loop_sleep_seconds", 1.0)),
            max_context_messages=int(agent_raw.get("max_context_messages", 20)),
            max_message_chars=int(agent_raw.get("max_message_chars", 4000)),
        ),
        execution=ExecutionConfig(
            shell=str(execution_raw.get("shell", "/bin/bash")),
            timeout_seconds=int(execution_raw.get("timeout_seconds", 120)),
            max_output_chars=int(execution_raw.get("max_output_chars", 8000)),
        ),
    )


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_template_file(path: Path) -> Path:
    return resolve_tool_root_dir() / "src" / "aibot" / path


def load_system_prompt(project_dir: Path) -> str:
    project_prompt = project_dir / "Prompts.md"
    if project_prompt.exists():
        return read_text_file(project_prompt)
    return read_text_file(resolve_template_file(DEFAULT_SYSTEM_PROMPT_RELATIVE_PATH))


def load_runtime_guardrails_prompt() -> str:
    return read_text_file(resolve_template_file(DEFAULT_GUARDRAILS_PROMPT_RELATIVE_PATH))


def resolve_tool_root_dir() -> Path:
    env_override = os.environ.get("AIBOT_TOOL_ROOT_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    module_dir = Path(__file__).resolve().parent
    for candidate in [module_dir, module_dir.parent, module_dir.parent.parent]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return module_dir.parent


def resolve_tool_skills_dir() -> Path | None:
    env_override = os.environ.get("AIBOT_TOOL_SKILLS_DIR")
    if env_override:
        candidate = Path(env_override).expanduser().resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate

    tool_root = resolve_tool_root_dir()
    candidates = [tool_root / "skills"]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def load_skills_context(project_dir: Path) -> tuple[str, list[str]]:
    skill_roots = [
        project_dir / "skills",
        Path.home() / ".codex" / "skills",
        Path.home() / ".config" / "aibot" / "skills",
    ]

    tool_skills_dir = resolve_tool_skills_dir()
    if tool_skills_dir is not None:
        skill_roots.append(tool_skills_dir)

    seen_roots: set[str] = set()
    deduped_roots: list[Path] = []
    for root in skill_roots:
        root_key = str(root.resolve()) if root.exists() else str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        deduped_roots.append(root)

    loaded_paths: list[str] = []
    sections: list[str] = []
    seen_paths: set[str] = set()

    for root in deduped_roots:
        if not root.exists() or not root.is_dir():
            continue

        for path in sorted(root.rglob("*.md")):
            if not path.is_file():
                continue

            resolved_key = str(path.resolve())
            if resolved_key in seen_paths:
                continue
            seen_paths.add(resolved_key)

            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue

            loaded_paths.append(str(path))
            sections.append(f"## Skill: {path}\n\n{content.strip()}")

    if not sections:
        return "", []

    combined = "\n\n".join(sections)
    return combined, loaded_paths


def read_initial_prompt(initial_prompt: str | None) -> str | None:
    if initial_prompt:
        return initial_prompt.strip() or None
    if not sys.stdin.isatty():
        data = sys.stdin.read().strip()
        if data:
            return data
    return None


def build_messages(
    system_prompt: str,
    guardrails_prompt: str,
    context_window_info: str,
    runtime_entries: list[dict[str, str]],
    max_context_messages: int,
) -> list[dict[str, str]]:
    context_entries = runtime_entries[-max_context_messages:] if max_context_messages > 0 else runtime_entries
    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": guardrails_prompt},
        {"role": "system", "content": context_window_info},
        *context_entries,
    ]


def resolve_context_window_tokens(model: str, configured_tokens: int) -> tuple[int, str]:
    if configured_tokens > 0:
        return configured_tokens, "config"

    model_key = model.strip().lower()
    from_exact = MODEL_CONTEXT_WINDOW_TOKENS.get(model_key)
    if from_exact is not None:
        return from_exact, "lookup_exact"

    for prefix, tokens in MODEL_CONTEXT_WINDOW_PREFIXES:
        if model_key.startswith(prefix):
            return tokens, "lookup_prefix"

    return 0, "unknown"


def make_context_window_info(
    state: ContextWindowState,
    max_context_messages: int,
    max_message_chars: int,
) -> str:
    configured_window = str(state.configured_window_tokens) if state.configured_window_tokens > 0 else "unknown"
    usage_ratio = "unknown"
    if state.configured_window_tokens > 0 and state.last_total_tokens > 0:
        usage_ratio = f"{(state.last_total_tokens / state.configured_window_tokens) * 100:.2f}%"

    estimated_char_budget = max_context_messages * max_message_chars
    return (
        "# Context Window Info\n"
        f"- model: {state.model}\n"
        f"- context_window_source: {state.source}\n"
        f"- configured_window_tokens: {configured_window}\n"
        f"- last_prompt_tokens: {state.last_prompt_tokens}\n"
        f"- last_completion_tokens: {state.last_completion_tokens}\n"
        f"- last_total_tokens: {state.last_total_tokens}\n"
        f"- last_usage_ratio: {usage_ratio}\n"
        f"- rolling_prompt_tokens: {state.rolling_prompt_tokens}\n"
        f"- rolling_completion_tokens: {state.rolling_completion_tokens}\n"
        f"- rolling_total_tokens: {state.rolling_total_tokens}\n"
        f"- max_context_messages: {max_context_messages}\n"
        f"- max_message_chars: {max_message_chars}\n"
        f"- estimated_context_char_budget: {estimated_char_budget}"
    )


def read_usage_tokens(response: Any) -> tuple[int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return prompt_tokens, completion_tokens, total_tokens


def extract_commands(content: str) -> list[str]:
    commands: list[str] = []
    in_block = False
    current_lines: list[str] = []

    for raw_line in content.splitlines():
        marker = raw_line.strip()

        if not in_block and marker == COMMAND_BLOCK_BEGIN:
            in_block = True
            current_lines = []
            continue

        if in_block and marker == COMMAND_BLOCK_END:
            command_text = "\n".join(current_lines).strip()
            if command_text:
                commands.append(command_text)
            in_block = False
            current_lines = []
            continue

        if in_block:
            current_lines.append(raw_line)

    return commands


def has_restart_meta_command(content: str) -> bool:
    in_block = False

    for raw_line in content.splitlines():
        marker = raw_line.strip()

        if not in_block and marker == COMMAND_BLOCK_BEGIN:
            in_block = True
            continue

        if in_block and marker == COMMAND_BLOCK_END:
            in_block = False
            continue

        if not in_block and marker.lower() == "restart":
            return True

    return False


def has_unquoted_heredoc(command: str) -> bool:
    index = 0
    length = len(command)

    while index < length - 1:
        if command[index] == "<" and command[index + 1] == "<":
            cursor = index + 2
            while cursor < length and command[cursor].isspace():
                cursor += 1

            if cursor < length and command[cursor] in {"'", '"'}:
                index = cursor + 1
                continue

            if cursor < length and (command[cursor].isalpha() or command[cursor] == "_"):
                return True

            index = cursor
            continue

        index += 1

    return False


def create_trace_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{os.getpid()}"
    trace_dir = base_dir / run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def append_trace_event(trace_dir: Path, event: dict[str, Any]) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    events_file = trace_dir / "events.jsonl"
    with events_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_trace_text(trace_dir: Path, name: str, content: str) -> None:
    (trace_dir / name).write_text(content, encoding="utf-8")


def run_command(command: str, execution: ExecutionConfig) -> CommandResult:
    if has_unquoted_heredoc(command):
        error_output = (
            f"$ {command}\n"
            "[error] rejected unsafe unquoted heredoc delimiter. "
            "Use quoted heredoc delimiters like <<'EOF' to prevent shell expansion."
        )
        return CommandResult(
            output=truncate_text(error_output, execution.max_output_chars),
            exit_code=2,
            timed_out=False,
        )

    try:
        completed = subprocess.run(
            [execution.shell],
            input=command,
            capture_output=True,
            text=True,
            timeout=execution.timeout_seconds,
            check=False,
        )
        output = (
            f"$ {command}\n"
            f"[exit_code] {completed.returncode}\n"
            f"[stdout]\n{completed.stdout}\n"
            f"[stderr]\n{completed.stderr}"
        )
        return CommandResult(
            output=truncate_text(output, execution.max_output_chars),
            exit_code=completed.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        timeout_output = (
            f"$ {command}\n"
            f"[error] timeout after {execution.timeout_seconds}s"
        )
        return CommandResult(
            output=truncate_text(timeout_output, execution.max_output_chars),
            exit_code=None,
            timed_out=True,
        )


def resolve_openrouter_api_key(config: OpenRouterConfig) -> str | None:
    if config.api_key_env.startswith("sk-or-"):
        return config.api_key_env
    return os.environ.get(config.api_key_env)


def resolve_project_dir(project_dir_arg: str) -> Path:
    return Path(project_dir_arg).expanduser().resolve()


def resolve_config_path(project_dir: Path, config_arg: str | None) -> tuple[Path | None, str]:
    if config_arg:
        candidate = Path(config_arg).expanduser()
        if candidate.is_absolute():
            return candidate, "explicit"
        return project_dir / candidate, "explicit"

    local_config = project_dir / "config.toml"
    if local_config.exists():
        return local_config, "project"

    home_config = Path.home() / ".config" / "aibot" / "config.toml"
    if home_config.exists():
        return home_config, "home"

    tool_config = resolve_tool_root_dir() / "config.toml"
    if tool_config.exists():
        return tool_config, "tool"

    return None, "default"


def ensure_runtime_scaffold(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "skills").mkdir(exist_ok=True)
    if not (project_dir / "skills" / ".gitkeep").exists():
        (project_dir / "skills" / ".gitkeep").write_text("", encoding="utf-8")
    if not (project_dir / "memory.md").exists():
        (project_dir / "memory.md").write_text("", encoding="utf-8")


def write_if_missing(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")


def write_agents_file(project_dir: Path, profile: str, force: bool) -> None:
    relative_template_path = AGENTS_PROFILE_TEMPLATE_MAP[profile]
    template_path = resolve_template_file(relative_template_path)
    template_content = read_text_file(template_path)
    write_if_missing(project_dir / "AGENTS.md", template_content, force)


def init_project(project_dir: Path, force: bool, agents_profile: str) -> int:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "skills").mkdir(exist_ok=True)

    write_if_missing(project_dir / "skills" / ".gitkeep", "", force)
    write_if_missing(project_dir / "memory.md", "", force)
    write_agents_file(project_dir, agents_profile, force)

    print(f"Initialized aibot project files in: {project_dir}")
    return 0


def run_agent(
    project_dir: Path,
    config_path: Path | None,
    config_source: str,
    initial_prompt_arg: str | None,
) -> int:
    ensure_runtime_scaffold(project_dir)
    os.chdir(project_dir)
    load_dotenv(project_dir / ".env")

    traces_root = project_dir / "traces"
    trace_dir = create_trace_dir(traces_root)
    append_trace_event(
        trace_dir,
        {
            "type": "run_started",
            "project_dir": str(project_dir),
            "config_source": config_source,
            "config_path": str(config_path) if config_path is not None else "",
        },
    )

    if config_path is None:
        config = load_default_config()
        config_snapshot_text = DEFAULT_CONFIG_TOML
    elif config_path.exists():
        config = load_config(config_path)
        config_snapshot_text = config_path.read_text(encoding="utf-8")
    else:
        append_trace_event(
            trace_dir,
            {
                "type": "startup_error",
                "error": f"Config file not found: {config_path}",
            },
        )
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1

    base_system_prompt = load_system_prompt(project_dir)
    runtime_guardrails_prompt = load_runtime_guardrails_prompt()

    skills_context, skill_paths = load_skills_context(project_dir)
    if skills_context:
        system_prompt = (
            f"{base_system_prompt}\n\n"
            "# Skills Context\n"
            "Loaded from project, home config, and tool source skill directories.\n\n"
            f"{skills_context}"
        )
    else:
        system_prompt = base_system_prompt

    append_trace_event(
        trace_dir,
        {
            "type": "skills_loaded",
            "count": len(skill_paths),
            "paths": skill_paths,
        },
    )

    write_trace_text(trace_dir, "skills_paths.txt", "\n".join(skill_paths) if skill_paths else "")
    write_trace_text(trace_dir, "system_prompt.md", system_prompt)
    write_trace_text(trace_dir, "config_snapshot.toml", config_snapshot_text)

    api_key = resolve_openrouter_api_key(config.openrouter)
    if not api_key:
        error_message = f"{config.openrouter.api_key_env} is not set"
        append_trace_event(trace_dir, {"type": "startup_error", "error": error_message})
        print(error_message, file=sys.stderr)
        return 1

    initial_prompt = read_initial_prompt(initial_prompt_arg)
    runtime_entries: list[dict[str, str]] = []
    context_window_tokens, context_window_source = resolve_context_window_tokens(
        config.openrouter.model,
        config.openrouter.context_window_tokens,
    )
    context_state = ContextWindowState(
        model=config.openrouter.model,
        source=context_window_source,
        configured_window_tokens=context_window_tokens,
        last_prompt_tokens=0,
        last_completion_tokens=0,
        last_total_tokens=0,
        rolling_prompt_tokens=0,
        rolling_completion_tokens=0,
        rolling_total_tokens=0,
    )
    append_trace_event(
        trace_dir,
        {
            "type": "context_window_resolved",
            "model": config.openrouter.model,
            "source": context_window_source,
            "context_window_tokens": context_window_tokens,
        },
    )

    effective_prompt = initial_prompt if initial_prompt else DEFAULT_PING_PROMPT
    truncated_prompt = truncate_text(effective_prompt, config.agent.max_message_chars)
    runtime_entries.append(
        {
            "role": "user",
            "content": truncated_prompt,
        }
    )
    write_trace_text(trace_dir, "initial_prompt.txt", truncated_prompt)
    append_trace_event(
        trace_dir,
        {
            "type": "initial_prompt_loaded",
            "chars": len(truncated_prompt),
            "mode": "request" if initial_prompt else "ping",
        },
    )

    client = OpenAI(base_url=config.openrouter.base_url, api_key=api_key)

    step = 0
    while True:
        if config.agent.max_steps > 0 and step >= config.agent.max_steps:
            append_trace_event(trace_dir, {"type": "run_finished", "reason": "max_steps"})
            break

        context_window_info = make_context_window_info(
            context_state,
            max_context_messages=config.agent.max_context_messages,
            max_message_chars=config.agent.max_message_chars,
        )
        messages = build_messages(
            system_prompt,
            runtime_guardrails_prompt,
            context_window_info,
            runtime_entries,
            config.agent.max_context_messages,
        )
        write_trace_text(
            trace_dir,
            f"step-{step:04d}-request.json",
            json.dumps(messages, indent=2, ensure_ascii=False),
        )

        try:
            response = client.chat.completions.create(
                model=config.openrouter.model,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            append_trace_event(
                trace_dir,
                {
                    "type": "openrouter_error",
                    "step": step,
                    "model": config.openrouter.model,
                    "error": str(exc),
                },
            )
            print(f"OpenRouter request failed: {exc}", file=sys.stderr)
            return 1

        prompt_tokens, completion_tokens, total_tokens = read_usage_tokens(response)
        context_state.last_prompt_tokens = prompt_tokens
        context_state.last_completion_tokens = completion_tokens
        context_state.last_total_tokens = total_tokens
        context_state.rolling_prompt_tokens += prompt_tokens
        context_state.rolling_completion_tokens += completion_tokens
        context_state.rolling_total_tokens += total_tokens
        append_trace_event(
            trace_dir,
            {
                "type": "context_usage",
                "step": step,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "rolling_total_tokens": context_state.rolling_total_tokens,
                "configured_window_tokens": context_state.configured_window_tokens,
            },
        )

        assistant_text = response.choices[0].message.content or ""
        if not isinstance(assistant_text, str):
            assistant_text = str(assistant_text)

        assistant_text = truncate_text(assistant_text, config.agent.max_message_chars)
        print(assistant_text)

        write_trace_text(trace_dir, f"step-{step:04d}-assistant.txt", assistant_text)
        append_trace_event(
            trace_dir,
            {
                "type": "assistant_response",
                "step": step,
                "chars": len(assistant_text),
            },
        )

        runtime_entries.append({"role": "assistant", "content": assistant_text})

        if has_restart_meta_command(assistant_text):
            append_trace_event(trace_dir, {"type": "meta_restart", "step": step})
            return 85

        commands = extract_commands(assistant_text)
        if not commands:
            runtime_entries.append(
                {
                    "role": "user",
                    "content": DEFAULT_NO_COMMAND_PROMPT,
                }
            )
            append_trace_event(trace_dir, {"type": "no_command", "step": step})
            step += 1
            if config.agent.loop_sleep_seconds > 0:
                time.sleep(config.agent.loop_sleep_seconds)
            continue

        for command_index, command in enumerate(commands):
            write_trace_text(trace_dir, f"step-{step:04d}-cmd-{command_index:02d}.sh", command + "\n")
            result = run_command(command, config.execution)
            print(result.output)

            write_trace_text(trace_dir, f"step-{step:04d}-cmd-{command_index:02d}.out", result.output)
            append_trace_event(
                trace_dir,
                {
                    "type": "command_executed",
                    "step": step,
                    "command_index": command_index,
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                },
            )

            runtime_entries.append(
                {
                    "role": "user",
                    "content": truncate_text(
                        f"Command execution result:\n{result.output}",
                        config.agent.max_message_chars,
                    ),
                }
            )

        step += 1
        if config.agent.loop_sleep_seconds > 0:
            time.sleep(config.agent.loop_sleep_seconds)

    return 0


def main() -> int:
    args = parse_args()

    if args.command == "init":
        project_dir = resolve_project_dir(args.project_dir)
        return init_project(project_dir, args.force, args.agents_profile)

    project_dir = resolve_project_dir(args.project_dir)
    config_path, config_source = resolve_config_path(project_dir, args.config)

    initial_prompt = args.initial_prompt
    while True:
        exit_code = run_agent(project_dir, config_path, config_source, initial_prompt)
        if exit_code != 85:
            return exit_code

        config_path, config_source = resolve_config_path(project_dir, args.config)
        initial_prompt = None


def main_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
