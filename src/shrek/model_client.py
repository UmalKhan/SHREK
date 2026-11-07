"""Backend-independent model calls, bounded call count, and auditable usage."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Protocol
import uuid
from urllib.error import HTTPError


class BudgetExceeded(RuntimeError):
    pass


class ModelOutputError(ValueError):
    pass


class CallBudget:
    """A shared attempt ceiling across inference, learning, and replay clients."""
    def __init__(self, maximum: int):
        if type(maximum) is not int or maximum < 1:
            raise ValueError("Total call budget must be a positive integer")
        self.maximum, self.used = maximum, 0

    def reserve(self) -> None:
        if self.used >= self.maximum:
            raise BudgetExceeded("Shared model-call budget exhausted")
        self.used += 1


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str = "unknown"


class Backend(Protocol):
    model_id: str
    is_mock: bool

    def complete(self, role: str, system: str, payload: dict, max_output_tokens: int) -> Completion: ...


class JsonlLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    try:
        value = json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (ValueError, TypeError) as error:
        raise ModelOutputError("Model did not return valid JSON") from error
    if not isinstance(value, dict):
        raise ModelOutputError("Model must return one JSON object")
    return value


class MeteredClient:
    def __init__(self, backend: Backend, log: JsonlLog, max_calls: int = 100, max_output_tokens: int = 384,
                 *, budget: CallBudget | None = None, purpose: str = "unspecified"):
        if type(max_calls) is not int or max_calls < 1 or type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Call and output limits must be positive integers")
        self.backend, self.log = backend, log
        self.max_calls, self.max_output_tokens = max_calls, max_output_tokens
        self.calls = 0
        self.budget, self.purpose = budget, purpose
        self.input_tokens = self.output_tokens = 0
        self.unknown_usage_calls = 0
        self.last_completion: Completion | None = None

    def ask(self, role: str, system: str, payload: dict, *, structured: bool = True, event_id: str = ""):
        if self.calls >= self.max_calls:
            raise BudgetExceeded("Model-call budget exhausted; no additional backend call made")
        if self.budget is not None:
            self.budget.reserve()
        # Reserve before attempting the call. Failures and invalid outputs still cost calls.
        self.calls += 1
        self.last_completion = None
        received = False
        started = time.perf_counter()
        record = {"call_id": uuid.uuid4().hex, "event_id": event_id, "role": role, "purpose": self.purpose, "model": self.backend.model_id,
                  "is_mock": self.backend.is_mock, "system": system, "payload": payload,
                  "max_output_tokens": self.max_output_tokens, "status": "error"}
        try:
            completion = self.backend.complete(role, system, payload, self.max_output_tokens)
            received = True
            self.last_completion = completion
            record["completion"] = asdict(completion)
            if completion.input_tokens is None or completion.output_tokens is None:
                self.unknown_usage_calls += 1
            else:
                self.input_tokens += completion.input_tokens
                self.output_tokens += completion.output_tokens
            if completion.finish_reason in {"length", "max_tokens"} and structured:
                raise ModelOutputError("Structured output was truncated")
            result = parse_object(completion.text) if structured else completion.text
            record["status"] = "ok"
            return result
        except Exception as error:
            if not received:
                self.unknown_usage_calls += 1
            # Avoid serializing provider exception text, which can contain credentials/URLs.
            record["error_type"] = type(error).__name__
            if isinstance(error, HTTPError):
                record["http_status"] = error.code
            raise
        finally:
            record["latency_seconds"] = time.perf_counter() - started
            self.log.append(record)

    def usage(self) -> dict:
        return {"calls": self.calls, "max_calls": self.max_calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "unknown_usage_calls": self.unknown_usage_calls,
                "is_mock": self.backend.is_mock, "model": self.backend.model_id}


class ScriptedBackend:
    """Offline engineering fixture. Never a model-quality benchmark."""
    model_id = "scripted-fixture-v1"
    is_mock = True

    def __init__(self, steps: list[tuple[str, object]]):
        self.steps = iter(steps)

    def complete(self, role: str, system: str, payload: dict, max_output_tokens: int) -> Completion:
        expected, output = next(self.steps)
        if role != expected:
            raise AssertionError(f"Expected role {expected}, received {role}")
        if isinstance(output, Exception):
            raise output
        text = output if isinstance(output, str) else json.dumps(output)
        return Completion(text, finish_reason="stop")
