import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import API_KEY, BASE_URL, MODEL

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    parts: list[dict]


@dataclass
class Conversation:
    messages: list[Message] = field(default_factory=list)
    latest_image: bytes | None = None
    latest_text: str = ""


def image_to_b64(image_path: str | Path) -> str:
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode()


def b64_to_bytes(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)


def _get_mime_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
    return mime_map.get(suffix, "image/png")


def _build_image_part(image_path: str | Path) -> dict:
    return {
        "inline_data": {
            "mime_type": _get_mime_type(image_path),
            "data": image_to_b64(image_path),
        }
    }


def _build_image_part_from_bytes(image_bytes: bytes, mime_type: str = "image/png") -> dict:
    return {
        "inline_data": {
            "mime_type": mime_type,
            "data": base64.b64encode(image_bytes).decode(),
        }
    }


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=3, min=5, max=60))
def _call_api(payload: dict, response_modalities: list[str]) -> dict:
    url = f"{BASE_URL}/{MODEL}:generateContent"
    headers = {
        "x-goog-api-key": API_KEY,
        "Content-Type": "application/json",
    }

    gen_config = {
        "temperature": 1.0 if "image" in response_modalities else 0.2,
        "maxOutputTokens": 65536 if "image" in response_modalities else 4096,
        "responseModalities": response_modalities,
    }
    payload["generationConfig"] = gen_config

    logger.info(f"Calling Gemini API: {MODEL}, modalities={response_modalities}")
    with httpx.Client(timeout=180.0) as client:
        resp = client.post(url, headers=headers, json=payload)

    if resp.status_code != 200:
        logger.error(f"API error {resp.status_code}: {resp.text[:500]}")
        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:200]}")

    return resp.json()


def _parse_response(response: dict) -> tuple[bytes | None, str, list[dict]]:
    """Parse API response, return (image_bytes, text, raw_parts)."""
    candidates = response.get("candidates", [])
    if not candidates:
        logger.error(f"Full response: {json.dumps(response, ensure_ascii=False)[:1000]}")
        raise RuntimeError(f"No candidates in response: {json.dumps(response)[:300]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    image_bytes = None
    text = ""

    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline:
            image_bytes = b64_to_bytes(inline["data"])
        elif "text" in part:
            text += part["text"]

    return image_bytes, text, parts


def start_conversation(prompt: str, image_paths: list[str | Path]) -> Conversation:
    """Start a new generation conversation with initial prompt and images."""
    parts: list[dict] = [{"text": prompt}]
    for img_path in image_paths:
        parts.append(_build_image_part(img_path))

    user_msg = Message(role="user", parts=parts)
    payload = {"contents": [{"role": user_msg.role, "parts": user_msg.parts}]}

    response = _call_api(payload, response_modalities=["image", "text"])
    image_bytes, text, raw_parts = _parse_response(response)

    if image_bytes is None:
        raise RuntimeError("No image generated in response")

    model_msg = Message(role="model", parts=raw_parts)

    conv = Conversation()
    conv.messages = [user_msg, model_msg]
    conv.latest_image = image_bytes
    conv.latest_text = text
    return conv


def continue_conversation(conv: Conversation, prompt: str, image_paths: list[str | Path] | None = None) -> Conversation:
    """Continue an existing conversation with a fix instruction.

    Note: Do NOT re-send images in subsequent turns - Gemini's multi-turn
    requires thought_signature for all image parts after the first turn.
    Instead, reference images from the first turn via text.
    """
    parts: list[dict] = [{"text": prompt}]

    user_msg = Message(role="user", parts=parts)
    conv.messages.append(user_msg)

    contents = [{"role": m.role, "parts": m.parts} for m in conv.messages]
    payload = {"contents": contents}

    response = _call_api(payload, response_modalities=["image", "text"])
    image_bytes, text, raw_parts = _parse_response(response)

    if image_bytes is None:
        raise RuntimeError("No image generated in fix response")

    model_msg = Message(role="model", parts=raw_parts)
    conv.messages.append(model_msg)
    conv.latest_image = image_bytes
    conv.latest_text = text
    return conv


def call_vision(prompt: str, image_data_list: list[bytes]) -> str:
    """Call Gemini for text-only vision analysis (evaluation)."""
    parts: list[dict] = [{"text": prompt}]
    for img_bytes in image_data_list:
        parts.append(_build_image_part_from_bytes(img_bytes, "image/png"))

    payload = {"contents": [{"role": "user", "parts": parts}]}
    response = _call_api(payload, response_modalities=["text"])
    _, text, _ = _parse_response(response)
    return text
