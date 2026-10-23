import os
import json
import time
import base64
from urllib.parse import quote

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)


# Allow browser requests from Gyan Quest hosting domains.
ALLOWED_ORIGINS = {
    "https://gyanquest-edu.web.app",
    "https://gyanquest-edu.firebaseapp.com",
    "https://gyanquest-edu.ai.studio",
}


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")

    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

    return response


HF_SPACE = "https://krea-krea-2.hf.space"
# Keep Krea's own deadline short enough to leave time for the backup provider
# within the frontend's existing 75-second request timeout.
KREA_TIMEOUT_SECONDS = 45
POLLINATIONS_TIMEOUT_SECONDS = max(
    5, min(int(os.environ.get("POLLINATIONS_TIMEOUT_SECONDS", "25")), 60)
)
POLLINATIONS_URL = "https://gen.pollinations.ai/image/"


@app.get("/")
def home():
    return jsonify({
        "service": "Gyan Quest Image API",
        "status": "online"
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


def generate_with_krea(prompt):
    """Try the existing Krea provider, with a bounded overall deadline."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not configured")

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, watermark, unwanted text",
        "model": "Turbo",
        "steps": 8,
        "guidance": 0.0,
        "width": 1024,
        "height": 1024,
        "seed": 0,
        "randomize": True
    }
    deadline = time.monotonic() + KREA_TIMEOUT_SECONDS

    response = requests.post(
        f"{HF_SPACE}/gradio_api/call/v2/generate",
        headers=headers,
        json=payload,
        timeout=(10, min(15, KREA_TIMEOUT_SECONDS))
    )
    response.raise_for_status()
    event_id = response.json()["event_id"]

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise requests.Timeout("Krea generation exceeded its time limit")

    result = requests.get(
        f"{HF_SPACE}/gradio_api/call/generate/{event_id}",
        headers=headers,
        stream=True,
        timeout=(10, max(1, remaining))
    )
    result.raise_for_status()

    for line in result.iter_lines(decode_unicode=True):
        if time.monotonic() > deadline:
            raise requests.Timeout("Krea generation exceeded its time limit")
        if not line or not line.startswith("data:"):
            continue

        content = line[5:].strip()
        try:
            output = json.loads(content)
        except json.JSONDecodeError:
            continue

        if isinstance(output, list) and output:
            image_data = output[0]
            if isinstance(image_data, dict) and image_data.get("url"):
                return image_data["url"]

    raise ValueError("Krea returned no image URL")


def generate_with_pollinations(prompt):
    """Generate a backup image and return it as a data URL, keeping the key server-side."""
    api_key = os.environ.get("POLLINATIONS_API_KEY")
    if not api_key:
        raise RuntimeError("POLLINATIONS_API_KEY is not configured")

    model = os.environ.get("POLLINATIONS_IMAGE_MODEL", "flux")
    response = requests.get(
        f"{POLLINATIONS_URL}{quote(prompt, safe='')}",
        params={"model": model, "width": 1024, "height": 1024},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=(10, POLLINATIONS_TIMEOUT_SECONDS)
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
    if not content_type.startswith("image/") or not response.content:
        raise ValueError("Pollinations did not return image data")

    # The existing frontend expects { imageUrl }. Returning a data URL preserves
    # that contract without exposing the secret API key in a browser-visible URL.
    encoded_image = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded_image}"


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "A prompt is required"}), 400

    prompt = prompt.strip()

    try:
        image_url = generate_with_krea(prompt)
        return jsonify({"imageUrl": image_url, "provider": "krea"}), 200
    except Exception:
        app.logger.exception("Krea generation failed; trying Pollinations fallback")

    try:
        image_url = generate_with_pollinations(prompt)
        return jsonify({"imageUrl": image_url, "provider": "pollinations"}), 200
    except Exception:
        app.logger.exception("Pollinations fallback failed")
        return jsonify({
            "error": "Image generation failed with both Krea and Pollinations. Check server logs and provider configuration."
        }), 502
