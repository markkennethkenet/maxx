"""
MAXX backend — receives a selfie from the Flutter app, sends it to
OpenAI's Vision model with a strict JSON schema (Structured Outputs),
and returns a clean ScanResult JSON. The OpenAI API key lives ONLY here,
never in the Flutter app.

Run:
    pip install flask flask-cors openai python-dotenv --break-system-packages
    export OPENAI_API_KEY=sk-...
    python app.py
"""

import base64
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8MB cap

# Structured Outputs schema — the model MUST return exactly this shape.
SCAN_RESULT_SCHEMA = {
    "name": "scan_result",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "profile_score": {"type": "integer"},
            "lighting_score": {"type": "integer"},
            "lighting_label": {"type": "string"},
            "angle_score": {"type": "integer"},
            "angle_label": {"type": "string"},
            "photo_quality_score": {"type": "integer"},
            "photo_quality_label": {"type": "string"},
            "background_score": {"type": "integer"},
            "background_label": {"type": "string"},
            "grooming_score": {"type": "integer"},
            "grooming_label": {"type": "string"},
            "profile_readiness": {"type": "string"},
            "summary": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6,
            },
        },
        "required": [
            "profile_score", "lighting_score", "lighting_label",
            "angle_score", "angle_label", "photo_quality_score",
            "photo_quality_label", "background_score", "background_label",
            "grooming_score", "grooming_label", "profile_readiness",
            "summary", "recommendations",
        ],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """You are MAXX's profile-photo quality analyst.
Evaluate ONLY technical photo and presentation quality — lighting, camera
angle, image clarity, background, and general grooming/presentation
(hair neatness, visible cleanliness, appropriate attire framing).

Do NOT comment on or score: attractiveness, facial features, body shape,
race, ethnicity, age, gender presentation, or any physical trait unrelated
to photography technique. Never produce a "beauty" or "attractiveness"
rating. If the image does not clearly show a single person's face, or is
inappropriate, unclear, or not a real photo of a person, set all scores to
0, profile_readiness to "Not usable", and explain why in summary and give
a recommendation to retake the photo.

Scores are 0-100. Recommendations should be short, actionable, and about
photography/presentation only (e.g. lighting, angle, background, framing).
"""


def _moderate(image_data_url: str) -> tuple[bool, str]:
    """Run the image through OpenAI's moderation endpoint before analysis."""
    try:
        result = client.moderations.create(
            model="omni-moderation-latest",
            input=[{"type": "image_url", "image_url": {"url": image_data_url}}],
        )
        flagged = result.results[0].flagged
        if flagged:
            return False, "Image flagged by moderation and cannot be processed."
        return True, ""
    except Exception as e:
        # Fail closed: if moderation itself errors, don't proceed to analysis.
        return False, f"Moderation check failed: {e}"


@app.route("/scan", methods=["POST"])
def scan():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected field 'image')."}), 400

    file = request.files["image"]
    image_bytes = file.read()

    if len(image_bytes) == 0:
        return jsonify({"error": "Empty image file."}), 400
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return jsonify({"error": "Image too large (max 8MB)."}), 400

    mime = file.mimetype or "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    # 1. Moderation gate
    ok, reason = _moderate(data_url)
    if not ok:
        return jsonify({"error": reason}), 422

    # 2. Vision analysis with structured output
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this profile photo for MAXX.",
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            response_format={"type": "json_schema", "json_schema": SCAN_RESULT_SCHEMA},
            max_tokens=600,
        )
        result = response.choices[0].message.content
    except Exception as e:
        return jsonify({"error": f"Vision analysis failed: {e}"}), 502

    # response.choices[0].message.content is already a JSON string
    # matching SCAN_RESULT_SCHEMA — pass it straight through.
    return app.response_class(result, mimetype="application/json")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
