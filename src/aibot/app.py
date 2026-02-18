from __future__ import annotations

import argparse
import json
import os
import re
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

COMMAND_PATTERN = re.compile(r"<command>(.*?)</command>", re.DOTALL | re.IGNORECASE)


@dataclass(slots=True)
class OpenRouterConfig:
    model: str
    base_url: str
    api_key_env: str


@dataclass(slots=True)
class PromptConfig:
    system_file: str


@dataclass(slots=True)
class AgentConfig:
    max_steps: int
    loop_sleep_seconds: float
    max_context_messages: int
    max_message_chars: int
    no_command_prompt: str
    ping_prompt: str


@dataclass(slots=True)
class ExecutionConfig:
    shell: str
    timeout_seconds: int
    max_output_chars: int


@dataclass(slots=True)
class AppConfig:
    openrouter: OpenRouterConfig
    prompt: PromptConfig
    agent: AgentConfig
    execution: ExecutionConfig


@dataclass(slots=True)
class CommandResult:
    output: str
    exit_code: int | None
    timed_out: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous OpenRouter command harness")
    parser.add_argument(
        "initial_prompt",
        nargs="?",
        default=None,
        help="Initial user prompt. If omitted, reads from stdin if available.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to runtime config file.",
    )
    return parser.parse_args()


def truncate_text(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n...[truncated]"


def load_config(path: str) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    openrouter_raw = raw.get("openrouter", {})
    prompt_raw = raw.get("prompt", {})
    agent_raw = raw.get("agent", {})
    execution_raw = raw.get("execution", {})

    return AppConfig(
        openrouter=OpenRouterConfig(
            model=str(openrouter_raw.get("model", "openai/gpt-4o-mini")),
            base_url=str(openrouter_raw.get("base_url", "https://openrouter.ai/api/v1")),
            api_key_env=str(openrouter_raw.get("api_key_env", "OPENROUTER_API_KEY")),
        ),
        prompt=PromptConfig(
            system_file=str(prompt_raw.get("system_file", "Prompts.md")),
        ),
        agent=AgentConfig(
            max_steps=int(agent_raw.get("max_steps", 0)),
            loop_sleep_seconds=float(agent_raw.get("loop_sleep_seconds", 1.0)),
            max_context_messages=int(agent_raw.get("max_context_messages", 20)),
            max_message_chars=int(agent_raw.get("max_message_chars", 4000)),
            no_command_prompt=str(
                agent_raw.get(
                    "no_command_prompt",
                    "No command was detected. Continue by returning one or more commands using <command>...</command>.",
                )
            ),
            ping_prompt=str(
                agent_raw.get(
                    "ping_prompt",
                    "Ping: continue autonomous progress and output next actionable step.",
                )
            ),
        ),
        execution=ExecutionConfig(
            shell=str(execution_raw.get("shell", "/bin/bash")),
            timeout_seconds=int(execution_raw.get("timeout_seconds", 120)),
            max_output_chars=int(execution_raw.get("max_output_chars", 8000)),
        ),
    )


def read_text_file(path: str) -> str:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    return target.read_text(encoding="utf-8")


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
    runtime_entries: list[dict[str, str]],
    max_context_messages: int,
) -> list[dict[str, str]]:
    context_entries = runtime_entries[-max_context_messages:] if max_context_messages > 0 else runtime_entries
    return [{"role": "system", "content": system_prompt}, *context_entries]


def extract_commands(content: str) -> list[str]:
    commands = [command.strip() for command in COMMAND_PATTERN.findall(content)]
    return [command for command in commands if command]


def create_trace_dir(base_dir: str = "traces") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{os.getpid()}"
    trace_dir = Path(base_dir) / run_id
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
    try:
        completed = subprocess.run(
            command,
            shell=True,
            executable=execution.shell,
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


def main() -> int:
    args = parse_args()
    load_dotenv()

    trace_dir = create_trace_dir()
    append_trace_event(
        trace_dir,
        {
            "type": "run_started",
            "config_path": args.config,
        },
    )

    try:
        config = load_config(args.config)
        system_prompt = read_text_file(config.prompt.system_file)
    except FileNotFoundError as exc:
        append_trace_event(trace_dir, {"type": "startup_error", "error": str(exc)})
        print(str(exc), file=sys.stderr)
        return 1

    write_trace_text(trace_dir, "system_prompt.md", system_prompt)
    write_trace_text(trace_dir, "config_snapshot.toml", Path(args.config).read_text(encoding="utf-8"))

    api_key = resolve_openrouter_api_key(config.openrouter)
    if not api_key:
        error_message = f"{config.openrouter.api_key_env} is not set"
        append_trace_event(trace_dir, {"type": "startup_error", "error": error_message})
        print(error_message, file=sys.stderr)
        return 1

    initial_prompt = read_initial_prompt(args.initial_prompt)
    runtime_entries: list[dict[str, str]] = []

    effective_prompt = initial_prompt if initial_prompt else config.agent.ping_prompt
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

        messages = build_messages(system_prompt, runtime_entries, config.agent.max_context_messages)
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

        commands = extract_commands(assistant_text)
        if not commands:
            runtime_entries.append(
                {
                    "role": "user",
                    "content": config.agent.no_command_prompt,
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


def main_entry() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    main_entry()
