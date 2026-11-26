"""Optional LLM access. An LLM here is any callable `(system, user) -> str`.

`openai_compatible()` builds one for any OpenAI-compatible chat completions
endpoint, using only the standard library. The framework runs without an LLM:
the predictor falls back to a statistical rule and the code generator to a
template. Tests use plain Python functions in place of a model.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

LLM = Callable[[str, str], str]


def openai_compatible(base_url: str | None = None, api_key: str | None = None,
                      model: str | None = None, temperature: float = 0.3,
                      max_tokens: int = 800, timeout: int = 120) -> LLM:
    base_url = (base_url or os.environ["SEP_LLM_BASE_URL"]).rstrip("/")
    api_key = api_key or os.environ["SEP_LLM_API_KEY"]
    model = model or os.environ["SEP_LLM_MODEL"]
    if not base_url.startswith("https://"):
        raise ValueError("the LLM endpoint must use https")

    def call(system: str, user: str) -> str:
        body = json.dumps({
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urllib.request.Request(
            base_url + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)["choices"][0]["message"]["content"]

    return call


def strip_code_fences(text: str) -> str:
    """Models often wrap code in Markdown fences; remove the fence lines."""
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip("\n")
