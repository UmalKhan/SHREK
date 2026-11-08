"""Optional real-model backends. Importing this module does not load a model."""

import json
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from shrek.model_client import Completion, ModelOutputError


class CompatibleHTTPBackend:
    """Chat-completions interface for a local server or explicitly configured provider."""
    is_mock = False

    def __init__(self, base_url: str, model_id: str, api_key: str | None = None, timeout: float = 120,
                 rate_limit_retries: int = 0, retry_wait: float = 10.0):
        url = urlsplit(base_url)
        if not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Base URL must not contain credentials, query strings, or fragments")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ValueError("Use HTTPS except for localhost model servers")
        if not model_id.strip() or timeout <= 0:
            raise ValueError("Model ID and positive timeout are required")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_id = model_id
        self.api_key = api_key
        self.timeout = timeout
        # Only HTTP 429 (request rejected before processing) is retried; it is not a model attempt.
        self.rate_limit_retries, self.retry_wait = rate_limit_retries, retry_wait
        self.rate_limit_waits = 0

    def metadata(self) -> dict:
        return {"backend": "compatible-http", "model": self.model_id, "temperature": 0,
                "limitation": "Remote model revision/seed support must be recorded separately by the operator."}

    def complete(self, role: str, system: str, payload: dict, max_output_tokens: int) -> Completion:
        body = {"model": self.model_id, "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                "temperature": 0, "max_tokens": max_output_tokens}
        # Some providers' edge proxies reject the default Python-urllib agent (Cloudflare 1010).
        headers = {"Content-Type": "application/json", "User-Agent": "shrek-research/0.1"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.url, data=json.dumps(body).encode(), headers=headers, method="POST")
        # No automatic retries of processed requests: each attempt is accounted for explicitly.
        # Rate-limit rejections (429) are retried only when explicitly enabled, and counted.
        for attempt in range(self.rate_limit_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
                break
            except HTTPError as error:
                if error.code != 429 or attempt == self.rate_limit_retries:
                    raise
                self.rate_limit_waits += 1
                time.sleep(self.retry_wait * (attempt + 1))
        try:
            choice = result["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty completion")
            usage = result.get("usage") or {}
            def count(key):
                value = usage.get(key)
                return value if type(value) is int and value >= 0 else None
            return Completion(text, count("prompt_tokens"), count("completion_tokens"), choice.get("finish_reason", "unknown"))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ModelOutputError("Invalid chat-completions response") from error


JUDGE_ROLES = frozenset({"judge", "validate_outcome"})


class RoleRoutingBackend:
    """Sends the evaluator roles (online judge, gate outcome assessor) to a separate model.

    Every other role, including the protected host, stays on the primary (local) model.
    Tokenizer-based capacity accounting and seeding always refer to the primary model.
    """

    def __init__(self, primary, judge, judge_roles=JUDGE_ROLES):
        if not judge_roles:
            raise ValueError("At least one routed role is required")
        self.primary, self.judge, self.judge_roles = primary, judge, frozenset(judge_roles)
        self.model_id = primary.model_id
        self.is_mock = bool(primary.is_mock or judge.is_mock)

    def model_for(self, role: str) -> str:
        return self.judge.model_id if role in self.judge_roles else self.primary.model_id

    def metadata(self) -> dict:
        describe = lambda backend: backend.metadata() if hasattr(backend, "metadata") else {"model": backend.model_id}
        return {"backend": "role-routing", "model": self.primary.model_id, "primary": describe(self.primary),
                "judge": describe(self.judge), "judge_roles": sorted(self.judge_roles)}

    def __getattr__(self, name):
        # count_tokens, reset_seed, etc. belong to the primary (host) model.
        if name in {"count_tokens", "reset_seed"}:
            return getattr(self.primary, name)
        raise AttributeError(name)

    def complete(self, role: str, system: str, payload: dict, max_output_tokens: int) -> Completion:
        target = self.judge if role in self.judge_roles else self.primary
        return target.complete(role, system, payload, max_output_tokens)


class TransformersBackend:
    """One local frozen model, greedy decoding, no silent prompt truncation."""
    is_mock = False

    def __init__(self, model_id: str, revision: str = "main", device: str = "auto", context_limit: int = 4096, seed: int = 17,
                 load_in_4bit: bool = False):
        if context_limit < 128 or device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Invalid local model settings")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError('Install the optional local-model dependencies: pip install -e ".[hf]"') from error
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        self.torch = torch
        self.model_id, self.revision = model_id, revision
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.context_limit, self.seed = context_limit, seed
        self.load_in_4bit = load_in_4bit
        if load_in_4bit and self.device != "cuda":
            raise RuntimeError("4-bit loading requires a CUDA device")
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        extra = {}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as error:
                raise RuntimeError("4-bit loading needs bitsandbytes: pip install bitsandbytes") from error
            # NF4 weights with FP16 compute (Turing GPUs such as the RTX 20 series lack fast bfloat16).
            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                              bnb_4bit_compute_dtype=torch.float16,
                                                              bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=False,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device, **extra,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        self.context_limit = min(context_limit, getattr(self.model.config, "max_position_embeddings", context_limit))

    def metadata(self) -> dict:
        return {"backend": "transformers", "model": self.model_id, "requested_revision": self.revision,
                "resolved_revision": self.resolved_revision, "device": self.device, "seed": self.seed,
                "context_limit": self.context_limit, "do_sample": False, "weights_frozen": True,
                "quantization": "bnb-nf4-4bit-fp16-compute" if self.load_in_4bit else "none-fp16"}

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def reset_seed(self, seed: int) -> None:
        self.seed = seed
        self.torch.manual_seed(seed)

    def complete(self, role: str, system: str, payload: dict, max_output_tokens: int) -> Completion:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        input_length = inputs["input_ids"].shape[-1]
        if input_length + max_output_tokens > self.context_limit:
            raise ValueError("Prompt plus output budget exceeds context limit; shorten inputs/rules explicitly")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_output_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        generated = output[0][input_length:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        finished = len(generated) > 0 and int(generated[-1]) in eos_ids
        reason = "stop" if finished or len(generated) < max_output_tokens else "length"
        return Completion(self.tokenizer.decode(generated, skip_special_tokens=True), input_length, len(generated), reason)
