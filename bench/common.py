"""Shared client helpers for the GB10 benchmark scripts.

Endpoint and key come from the environment so nothing is baked in:

    export GB10_BASE_URL=http://127.0.0.1:18300/v1   # `./flash serve` default
    export GB10_MODEL=qwen3.8-flash-next

Against a proxy or gateway in front of the server, point it there instead and
set the key:

    export GB10_BASE_URL=http://127.0.0.1:8000/v1
    export GB10_API_KEY=<key>

Thinking is toggled through the checkpoint's `reasoning_effort` template
argument (this template takes xhigh/medium/low; the default serve recipe also
aliases high/max and minimal). This is the vLLM analog of the SGLang recipes'
`enable_thinking`; override the two values with GB10_REASONING_OFF /
GB10_REASONING_ON if you change the template.
"""
import json
import os
import time
import urllib.request

BASE_URL = os.environ.get("GB10_BASE_URL", "http://127.0.0.1:18300/v1").rstrip("/")
API_KEY = os.environ.get("GB10_API_KEY", "")
MODEL = os.environ.get("GB10_MODEL", "qwen3.8-flash-next")
REASONING_OFF = os.environ.get("GB10_REASONING_OFF", "low")
REASONING_ON = os.environ.get("GB10_REASONING_ON", "xhigh")

CHAT_URL = f"{BASE_URL}/chat/completions"
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

# The canonical decode-throughput prompt used for every figure in the README.
CODE_PROMPT = (
    "Write a complete Python implementation of an LRUCache class with get and put "
    "in O(1), using a dict and a doubly linked list. Include docstrings."
)


def chat(prompt, max_tokens, thinking=False, stream=False, timeout=3600):
    """One chat completion.

    Returns a dict with e2e seconds, completion/prompt token counts, and (when
    streaming) ttft. Always count `completion_tokens` over wall time rather than
    counting stream events: speculative decoding emits several tokens per event,
    so event-counting inflates the rate by roughly the acceptance length.
    """
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {
            "reasoning_effort": REASONING_ON if thinking else REASONING_OFF,
        },
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(CHAT_URL, json.dumps(body).encode(), HEADERS)
    t0 = time.time()

    if not stream:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        usage = d["usage"]
        return {
            "e2e": time.time() - t0,
            "ttft": None,
            "completion_tokens": usage["completion_tokens"],
            "prompt_tokens": usage["prompt_tokens"],
            "finish_reason": d["choices"][0].get("finish_reason"),
            "content": d["choices"][0]["message"].get("content") or "",
        }

    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            choices = d.get("choices") or []
            if choices and ttft is None:
                delta = choices[0].get("delta", {}) or {}
                # vLLM's qwen3 reasoning parser uses "reasoning"; SGLang and the
                # OpenAI reasoning API use "reasoning_content".
                if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                    ttft = time.time() - t0
            if d.get("usage"):
                completion_tokens = d["usage"].get("completion_tokens", completion_tokens)
                prompt_tokens = d["usage"].get("prompt_tokens", prompt_tokens)
    return {
        "e2e": time.time() - t0,
        "ttft": ttft,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "finish_reason": None,
        "content": "",
    }
