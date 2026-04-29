"""Inference backend for Ollama."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


class BackendError(RuntimeError):
    """Raised when a backend operation fails."""


@dataclass
class BackendConfig:
    provider: str
    api_base: str
    temperature: float
    startup_wait_s: int = 30
    request_timeout_s: int = 600


def detect_provider(api_base: str) -> str:
    """Detect whether Ollama is available."""
    try:
        response = requests.get(f"{api_base.rstrip('/')}/api/tags", timeout=3)
        if response.ok:
            return "ollama"
    except requests.RequestException:
        pass

    raise BackendError(
        f"Could not detect Ollama at {api_base}. Expected /api/tags to respond."
    )


class InferenceBackend:
    """Shared interface for experiment runners."""

    provider = "unknown"

    def __init__(self, config: BackendConfig):
        self.config = config

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        return {
            "family": self.provider,
            "parameter_size": "unknown",
            "quantization_level": "unknown",
            "architecture": self.provider,
            "parameter_count": None,
            "embedding_length": None,
            "block_count": None,
            "feed_forward_length": None,
            "attention_head_count": None,
            "attention_head_count_kv": None,
            "attention_key_length": None,
            "attention_value_length": None,
            "attention_type": "unknown",
            "context_length": None,
            "vocab_size": None,
            "is_moe": None,
            "expert_count": None,
            "expert_used_count": None,
            "embedding_params": None,
            "attention_params": None,
            "ffn_params": None,
        }

    def warm_model(self, model_name: str) -> bool:
        return True

    def cold_reset(self, model_name: str) -> bool:
        return False

    def supports_cold_cache(self) -> bool:
        return False

    def run_inference(self, model_name: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
        raise NotImplementedError


class OllamaBackend(InferenceBackend):
    """Ollama-backed inference."""

    provider = "ollama"

    def list_models(self) -> list[str]:
        response = requests.get(
            f"{self.config.api_base.rstrip('/')}/api/tags",
            timeout=5,
        )
        response.raise_for_status()
        return [model["name"] for model in response.json().get("models", [])]

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.config.api_base.rstrip('/')}/api/show",
            json={"name": model_name, "verbose": True},
            timeout=10,
        )
        response.raise_for_status()
        info = response.json()
        details = info.get("details", {})
        model_info = info.get("model_info", {})
        arch = model_info.get("general.architecture", "unknown")

        def _get(key: str, default: Any = None) -> Any:
            return model_info.get(f"{arch}.{key}", default)

        head_count = _get("attention.head_count")
        head_count_kv = _get("attention.head_count_kv")
        if head_count is not None and head_count_kv is not None:
            if head_count_kv == 1:
                attention_type = "MQA"
            elif head_count_kv < head_count:
                attention_type = "GQA"
            else:
                attention_type = "MHA"
        else:
            attention_type = "unknown"

        expert_count = _get("expert_count")
        expert_used_count = _get("expert_used_count")
        is_moe = expert_count is not None and expert_count > 1

        embedding_length = _get("embedding_length")
        block_count = _get("block_count")
        ffn_length = _get("feed_forward_length")
        vocab_size = _get("vocab_size", model_info.get("llama.vocab_size"))

        embedding_params = None
        attention_params = None
        ffn_params = None
        if embedding_length and vocab_size:
            embedding_params = embedding_length * vocab_size
        if embedding_length and head_count and block_count:
            kv_dim = (head_count_kv or head_count) * (
                _get("attention.key_length") or (embedding_length // head_count)
            )
            attn_per_layer = (
                embedding_length * embedding_length
                + embedding_length * kv_dim
                + embedding_length * kv_dim
                + embedding_length * embedding_length
            )
            attention_params = attn_per_layer * block_count
        if embedding_length and ffn_length and block_count:
            ffn_per_layer = 3 * embedding_length * ffn_length
            if is_moe and expert_count:
                ffn_per_layer *= expert_count
            ffn_params = ffn_per_layer * block_count

        return {
            "family": details.get("family", "unknown"),
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization_level": details.get("quantization_level", "unknown"),
            "architecture": arch,
            "parameter_count": model_info.get("general.parameter_count"),
            "embedding_length": embedding_length,
            "block_count": block_count,
            "feed_forward_length": ffn_length,
            "attention_head_count": head_count,
            "attention_head_count_kv": head_count_kv,
            "attention_key_length": _get("attention.key_length"),
            "attention_value_length": _get("attention.value_length"),
            "attention_type": attention_type,
            "context_length": _get("context_length"),
            "vocab_size": vocab_size,
            "is_moe": is_moe,
            "expert_count": expert_count,
            "expert_used_count": expert_used_count,
            "embedding_params": embedding_params,
            "attention_params": attention_params,
            "ffn_params": ffn_params,
        }

    def warm_model(self, model_name: str) -> bool:
        try:
            requests.post(
                f"{self.config.api_base.rstrip('/')}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Warm up",
                    "options": {"num_predict": 1},
                    "stream": False,
                },
                timeout=60,
            ).raise_for_status()
            return True
        except requests.RequestException:
            return False

    def cold_reset(self, model_name: str) -> bool:
        try:
            requests.post(
                f"{self.config.api_base.rstrip('/')}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
                timeout=30,
            ).raise_for_status()
            time.sleep(2)
            return True
        except requests.RequestException:
            return False

    def supports_cold_cache(self) -> bool:
        return True

    def run_inference(self, model_name: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_output_tokens,
                "temperature": self.config.temperature,
                "top_p": 0.9,
            },
        }

        t0 = time.time()
        response = requests.post(
            f"{self.config.api_base.rstrip('/')}/api/generate",
            json=payload,
            timeout=self.config.request_timeout_s,
            stream=True,
        )
        response.raise_for_status()

        full_response = ""
        last_data: dict[str, Any] = {}
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                full_response += chunk["response"]
            if chunk.get("done"):
                last_data = chunk
        t1 = time.time()

        input_tokens = last_data.get("prompt_eval_count", 0) or max(1, len(prompt) // 4)
        output_tokens = last_data.get("eval_count", 0) or max(1, len(full_response) // 4)
        eval_ns = last_data.get("eval_duration", 0)
        tokens_per_sec = output_tokens / max(eval_ns / 1e9, 0.001) if eval_ns > 0 else output_tokens / max(t1 - t0, 0.001)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "wall_time_s": t1 - t0,
            "prompt_eval_ns": last_data.get("prompt_eval_duration", 0),
            "eval_ns": eval_ns,
            "load_ns": last_data.get("load_duration", 0),
            "tokens_per_sec": tokens_per_sec,
            "response_chars": len(full_response),
            "token_method": "api" if last_data.get("eval_count", 0) else "estimated",
        }


def build_backend(config: BackendConfig) -> InferenceBackend:
    """Instantiate the requested backend."""
    provider = config.provider
    if provider == "auto":
        provider = detect_provider(config.api_base)

    if provider == "ollama":
        return OllamaBackend(config)
    raise BackendError(f"Unsupported provider: {provider}")
