from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from shrek.backends import CompatibleHTTPBackend
from shrek.config import ExperimentConfig
from shrek.data import Example
from shrek.model_client import Completion, JsonlLog, MeteredClient, ModelOutputError
from shrek.runner import run_baseline


def test_http_adapter_through_full_runner_without_network_service(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert not self.headers["User-Agent"].startswith("Python-urllib")
            requests.append((self.path, payload))
            content = '{"label":"benign"}' if "Classify the structural" in payload["messages"][0]["content"] else "A safe test answer."
            response = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = CompatibleHTTPBackend(f"http://127.0.0.1:{server.server_port}/v1", "local-test-model")
        examples = [Example("a", "Explain clouds", "clouds", "adaptation", "fixture", "legitimate")]
        report = run_baseline(examples, backend, ExperimentConfig(), tmp_path / "http-test", learning=False)
        assert report["answered"] == 1 and report["events_with_errors"] == 0
        assert report["usage"]["input_tokens"] == 20 and report["usage"]["output_tokens"] == 10
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert all(item["max_tokens"] == 384 and item["temperature"] == 0 for _, item in requests)
        assert all("evaluation_label" not in json.dumps(item) for _, item in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("url", ["http://example.com/v1", "https://token@example.com/v1", "https://example.com/v1?key=secret", "ftp://example.com"])
def test_unsafe_or_credential_bearing_base_urls_rejected(url):
    with pytest.raises(ValueError):
        CompatibleHTTPBackend(url, "model")


def test_truncated_structured_output_is_not_accepted(tmp_path):
    class Backend:
        model_id = "truncation-fixture"
        is_mock = True
        def complete(self, *args):
            return Completion('{"label":"benign"}', 3, 4, "length")
    client = MeteredClient(Backend(), JsonlLog(tmp_path / "calls.jsonl"))
    with pytest.raises(ModelOutputError, match="truncated"):
        client.ask("classify", "system", {})
    assert client.calls == 1 and client.output_tokens == 4


class _Recorder:
    is_mock = False

    def __init__(self, model_id, text):
        self.model_id, self.text, self.roles = model_id, text, []

    def complete(self, role, system, payload, max_output_tokens):
        self.roles.append(role)
        return Completion(self.text, 1, 1, "stop")

    def count_tokens(self, text):
        return len(text.split())


def test_role_routing_sends_only_evaluator_roles_to_judge(tmp_path):
    from shrek.backends import RoleRoutingBackend
    local, remote = _Recorder("local-host", '{"ok":true}'), _Recorder("api-judge", '{"ok":true}')
    backend = RoleRoutingBackend(local, remote)
    client = MeteredClient(backend, JsonlLog(tmp_path / "calls.jsonl"))
    for role in ("classify", "host", "judge", "induce", "validate_outcome", "policy"):
        client.ask(role, "s", {}, structured=role != "host")
    assert remote.roles == ["judge", "validate_outcome"]
    assert local.roles == ["classify", "host", "induce", "policy"]
    logged = {json.loads(line)["role"]: json.loads(line)["model"] for line in (tmp_path / "calls.jsonl").read_text().splitlines()}
    assert logged["judge"] == "api-judge" and logged["host"] == "local-host"
    assert backend.count_tokens("a b c") == 3 and backend.model_id == "local-host"
    assert backend.metadata()["judge_roles"] == ["judge", "validate_outcome"]


def test_http_backend_retries_only_rate_limits():
    attempts = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            attempts.append(1)
            if len(attempts) < 3:
                self.send_response(429); self.send_header("Content-Length", "0"); self.end_headers(); return
            body = json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/v1"
        backend = CompatibleHTTPBackend(url, "m", rate_limit_retries=3, retry_wait=0)
        assert backend.complete("judge", "s", {}, 10).text == "{}"
        assert len(attempts) == 3 and backend.rate_limit_waits == 2
        attempts.clear()
        from urllib.error import HTTPError
        with pytest.raises(HTTPError):
            CompatibleHTTPBackend(url, "m").complete("judge", "s", {}, 10)
        assert len(attempts) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
