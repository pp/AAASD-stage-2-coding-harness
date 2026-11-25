"""Ollama chat client, cancellable requests, and tool-call extraction."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class ModelError(Exception):
    pass


class ModelCancelled(Exception):
    """The run was cancelled while a model request was in flight. The reply is discarded."""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_s: int, num_ctx: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx

    def check_model(self) -> None:
        try:
            with urllib.request.urlopen(self.base_url + "/api/tags", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError) as exc:
            raise ModelError("Ollama is not reachable at {}: {}".format(self.base_url, exc))
        names = {m.get("name") for m in data.get("models", [])}
        wanted = self.model if ":" in self.model else self.model + ":latest"
        if self.model not in names and wanted not in names:
            raise ModelError("Model '{}' is not installed. Run: ollama pull {}".format(self.model, self.model))

    def chat(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        body = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        req = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise ModelError("model request failed: {}".format(exc))
        message = data.get("message")
        if not isinstance(message, dict):
            raise ModelError("model reply had no message")
        return message

    def chat_cancellable(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]],
                         cancel: threading.Event, poll_s: float = 0.2) -> Dict[str, Any]:
        """Like chat(), but returns early with ModelCancelled when cancel is set.

        The request runs in a daemon thread. A cancelled request is abandoned: it has no side effects,
        and its reply is never used.
        """
        box: Dict[str, Any] = {}

        def worker() -> None:
            try:
                box["message"] = self.chat(messages, tools)
            except Exception as exc:  # passed back to the caller's thread
                box["error"] = exc

        thread = threading.Thread(target=worker, name="model-request", daemon=True)
        thread.start()
        while thread.is_alive():
            if cancel.is_set():
                raise ModelCancelled("model request cancelled")
            thread.join(poll_s)
        if "error" in box:
            raise box["error"]
        return box["message"]


def _as_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except ValueError:
        return None


def extract_tool_calls(message: Dict[str, Any]) -> Tuple[List[Tuple[Any, Any]], bool]:
    """Return ([(name, arguments)], from_text). Falls back to a JSON tool call written as text."""
    calls: List[Tuple[Any, Any]] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append((fn.get("name"), args))
    if calls:
        return calls, False
    parsed = _as_json(message.get("content") or "")
    if isinstance(parsed, dict) and isinstance(parsed.get("name"), str):
        args = parsed.get("arguments", parsed.get("parameters", {}))
        return [(parsed["name"], args)], True
    return [], False
