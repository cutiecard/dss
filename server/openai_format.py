"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array into one prompt, and wrap DeepSeek's
text output back into OpenAI response/stream objects.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterable, List

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Flatten a chat history into a single prompt DeepSeek can answer.

    A lone user message is sent verbatim. Multi-turn / system-prompted
    conversations are serialised with role labels and a trailing 'Assistant:'
    cue so the model continues in the right voice.
    """
    if len(messages) == 1 and messages[0].role == "user":
        return _text_of(messages[0].content)

    lines = []
    for m in messages:
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        lines.append(f"{label}: {_text_of(m.content)}")
    lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(model: str, content: str, prompt: str, reasoning: str | None = None) -> dict:
    pt, ct = _est_tokens(prompt), _est_tokens(content)
    rt = _est_tokens(reasoning) if reasoning else 0

    # Build the standard message object
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning  # DeepSeek extension

    return {
        "id": _id(),
        "object": "chat.completion",  # FIXED: Was "model_response"
        "created": _now(),
        "model": model,
        "choices": [                  # FIXED: Replaces "output" array
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "completion_tokens_details": {  # DeepSeek extension
                "reasoning_tokens": rt
            }
        }
    }


def stream_chunks(model: str, stream: Iterable[str]) -> Iterable[str]:
    cid, created = _id(), _now()
    accumulated_reasoning = []
    accumulated_content = []

    reasoning_open = False

    def frame(delta: dict, finish_reason=None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": "fp_44709d6fcb",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": None,
                    "finish_reason": finish_reason
                }
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame: announce role with empty content
    yield frame({"role": "assistant", "content": ""})

    for d in stream:
        if not d:
            continue

        if getattr(d, "kind", "message") == "reasoning":
            accumulated_reasoning.append(d.text)
            print(f"Reasoning chunk: {d.text}")
            if not reasoning_open:
                # Start thinking block
                yield frame({"content": "<think>" + d.text})
                reasoning_open = True
            else:
                yield frame({"content": d.text})

        else:
            # First normal content after reasoning: close thinking block
            if reasoning_open:
                yield frame({"content": "</think>" + d.text})
                reasoning_open = False
            else:
                yield frame({"content": d.text})
            print(f"Content chunk: {d.text}")
            accumulated_content.append(d.text)

    # If stream ends while still inside thinking, close it
    if reasoning_open:
        yield frame({"content": "</think>"})
        reasoning_open = False

    # Final stop chunk
    yield frame({}, finish_reason="stop")

    # Usage chunk
    pt = _est_tokens("")
    ct = _est_tokens(''.join(accumulated_content))
    rt = _est_tokens(''.join(accumulated_reasoning))

    usage_obj = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
            "completion_tokens_details": {"reasoning_tokens": rt}
        }
    }

    yield f"data: {json.dumps(usage_obj, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
