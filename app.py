
import os
import json
import time
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

HF_SPACE = "https://krea-krea-2.hf.space"
TIMEOUT = (15, 240)

@app.get("/")
def home():
    return jsonify({
        "service": "Gyan Quest Image API",
        "status": "online"
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"error": "A prompt is required"}), 400

    token = os.environ.get("HF_TOKEN")
    if not token:
        return jsonify({"error": "Server token is not configured"}), 500

    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "data": [
            prompt.strip(),
            "blurry, low quality, watermark, unwanted text",
            "Turbo",
            4,
            1,
            1024,
            1024,
            0,
            True
        ]
    }

    try:
        response = requests.post(
            f"{HF_SPACE}/gradio_api/call/v2/generate",
            headers=headers,
            json=payload,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        event_id = response.json()["event_id"]

        result = requests.get(
            f"{HF_SPACE}/gradio_api/call/generate/{event_id}",
            headers=headers,
            stream=True,
            timeout=TIMEOUT
        )
        result.raise_for_status()

        for line in result.iter_lines(decode_unicode=True):
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
                    return jsonify({
                        "imageUrl": image_data["url"]
                    }), 200

        return jsonify({"error": "No image URL in completed result"}), 502

    except requests.RequestException:
        app.logger.exception("Krea request failed")
        return jsonify({"error": "Krea 2 request failed"}), 502
    except (KeyError, ValueError, TypeError):
        app.logger.exception("Unexpected Krea response")
        return jsonify({"error": "Unexpected Krea response"}), 502
