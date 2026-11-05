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
