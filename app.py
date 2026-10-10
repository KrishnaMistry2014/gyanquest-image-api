
import os
import asyncio
from flask import Flask, request, jsonify
from gradio_client import Client

app = Flask(__name__)

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

    try:
        token = os.environ.get("HF_TOKEN")
        client = Client("krea/Krea-2", hf_token=token)

        result = client.predict(
            prompt=prompt.strip(),
            negative_prompt="blurry, low quality, text, watermark",
            model="Turbo",
            steps=4,
            guidance=1,
            width=1024,
            height=1024,
            seed=0,
            randomize=True,
            api_name="/generate"
        )

        return jsonify({
            "result": result
        }), 200

    except Exception:
        app.logger.exception("Krea generation failed")
        return jsonify({
            "error": "Image generation failed. Check the server logs."
        }), 502
