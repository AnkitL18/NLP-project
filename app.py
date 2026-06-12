from flask import Flask, render_template, request, jsonify
from lingua import LanguageDetectorBuilder
from transformers import MarianMTModel, MarianTokenizer
from dataset.home import translator
from sacrebleu import corpus_bleu
import json
import os
import gc

# ─────────────────────────────────────────────────────────────
# Flask App Initialization
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

print("======================================")
print("Starting NLP Translation Server...")
print("======================================")

# ─────────────────────────────────────────────────────────────
# Language Detector Initialization
# ─────────────────────────────────────────────────────────────

try:
    print("Loading language detector...")

    detector = (
        LanguageDetectorBuilder
        .from_all_languages()
        .with_preloaded_language_models()
        .build()
    )

    print("Language detector loaded successfully!")

except Exception as e:
    print("Error loading detector:", e)
    detector = None


# ─────────────────────────────────────────────────────────────
# Global Model Cache
# Stores already loaded Marian models
# ─────────────────────────────────────────────────────────────

model_cache = {}

# Limit cache size to prevent Render memory crash
MAX_MODELS_IN_MEMORY = 1


# ─────────────────────────────────────────────────────────────
# Supported Languages
# Keep fewer languages for Render free tier
# You can add more later
# ─────────────────────────────────────────────────────────────

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "fr": "French",
    "es": "Spanish",
    "de": "German"
}


# ─────────────────────────────────────────────────────────────
# Low Resource Languages
# These use your custom translator()
# instead of heavy MarianMT models
# ─────────────────────────────────────────────────────────────

LOW_RESOURCE = [
    "mr",
    "ta",
    "te",
    "gu",
    "pa",
    "bn",
    "ur",
    "sw",
    "th",
    "vi",
    "ms",
    "km",
    "lo",
    "my",
    "si",
    "ne",
    "hy",
    "ka"
]


# ─────────────────────────────────────────────────────────────
# Helper function to clear old models
# Prevents Render from running out of RAM
# ─────────────────────────────────────────────────────────────

def clear_model_cache():
    global model_cache

    print("Clearing old models from memory...")

    for _, model_data in model_cache.items():
        del model_data

    model_cache.clear()

    gc.collect()

    print("Memory cleaned.")
# ─────────────────────────────────────────────────────────────
# Load MarianMT Model Safely with Cache Control
# ─────────────────────────────────────────────────────────────

def load_model(model_name):
    global model_cache

    # Use existing model if already loaded
    if model_name in model_cache:
        print(f"Using cached model: {model_name}")
        return model_cache[model_name]

    try:
        print(f"Loading new model: {model_name}")

        # Free memory if cache limit reached
        if len(model_cache) >= MAX_MODELS_IN_MEMORY:
            clear_model_cache()

        tokenizer = MarianTokenizer.from_pretrained(
            model_name,
            local_files_only=False
        )

        model = MarianMTModel.from_pretrained(
            model_name,
            local_files_only=False
        )

        model_cache[model_name] = (tokenizer, model)

        print(f"Model loaded successfully: {model_name}")

        return tokenizer, model

    except Exception as e:
        print("======================================")
        print("MODEL LOADING FAILED")
        print("Model:", model_name)
        print("Error:", str(e))
        print("======================================")
        raise


# ─────────────────────────────────────────────────────────────
# Translation Helper
# ─────────────────────────────────────────────────────────────

def run_translation(text, model, tokenizer):

    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Reduced from 512 to save memory
        )

        output = model.generate(
            **inputs,
            max_length=128
        )

        translated_text = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return translated_text

    except Exception as e:
        print("Translation generation failed:", str(e))
        raise


# ─────────────────────────────────────────────────────────────
# Main Translation Logic
# ─────────────────────────────────────────────────────────────

def translate_text(text, src_code, tgt_code):

    print(
        f"Translation request: {src_code} -> {tgt_code}"
    )

    # Low resource languages use custom translator
    if src_code in LOW_RESOURCE or tgt_code in LOW_RESOURCE:

        print("Using custom low-resource translator")

        try:
            result = translator(
                text,
                src_code,
                tgt_code
            )

            return result, "Custom Translator"

        except Exception as e:
            print(
                "Custom translator failed:",
                str(e)
            )
            raise


    # Try direct Marian translation
    try:
        model_name = (
            f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        )

        tokenizer, model = load_model(model_name)

        result = run_translation(
            text,
            model,
            tokenizer
        )

        return result, "Marian MT"

    except Exception as e:

        print(
            "Direct translation failed:",
            str(e)
        )


    # Fallback: translate via English
    try:

        print("Trying English pivot translation")

        tokenizer1, model1 = load_model(
            f"Helsinki-NLP/opus-mt-{src_code}-en"
        )

        english_text = run_translation(
            text,
            model1,
            tokenizer1
        )


        tokenizer2, model2 = load_model(
            f"Helsinki-NLP/opus-mt-en-{tgt_code}"
        )

        final_text = run_translation(
            english_text,
            model2,
            tokenizer2
        )

        return final_text, "Marian MT Pivot"


    except Exception as e:

        print(
            "Pivot translation failed:",
            str(e)
        )


    # Final fallback
    try:

        print("Using final fallback translator")

        result = translator(
            text,
            src_code,
            tgt_code
        )

        return result, "Fallback Translator"

    except Exception as e:

        print(
            "All translation methods failed:",
            str(e)
        )

        raise Exception(
            "Translation service is currently unavailable"
        )
# ─────────────────────────────────────────────────────────────
# Home Route
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        languages=LANGUAGES
    )


# ─────────────────────────────────────────────────────────────
# Language Detection API
# ─────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received"
            }), 400

        text = data.get("text", "").strip()

        if not text:
            return jsonify({
                "error": "No text provided"
            }), 400


        if detector is None:
            return jsonify({
                "error": "Language detector not available"
            }), 500


        detected = detector.detect_language_of(text)


        if not detected:
            return jsonify({
                "error": "Could not detect language"
            }), 400


        return jsonify({
            "detected_code":
                detected.iso_code_639_1.name.lower(),

            "detected_name":
                detected.name
        })


    except Exception as e:

        print("DETECTION ERROR:", str(e))

        return jsonify({
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────────────────────
# Translation API
# ─────────────────────────────────────────────────────────────

@app.route("/translate", methods=["POST"])
def translate():

    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON received"
            }), 400


        text = data.get("text", "").strip()
        src_code = data.get("src_code", "")
        tgt_code = data.get("tgt_code", "")


        if not text or not src_code or not tgt_code:
            return jsonify({
                "error": "Missing translation fields"
            }), 400


        if src_code == tgt_code:
            return jsonify({
                "error": "Source and target languages cannot be the same"
            }), 400


        print("=" * 50)
        print("TRANSLATION REQUEST RECEIVED")
        print("Source:", src_code)
        print("Target:", tgt_code)
        print("Text:", text[:100])
        print("=" * 50)


        translated_text, engine = translate_text(
            text,
            src_code,
            tgt_code
        )


        return jsonify({
            "translated": translated_text,
            "engine": engine,
            "status": "success"
        })


    except Exception as e:

        print("=" * 50)
        print("TRANSLATION ERROR")
        print(str(e))
        print("=" * 50)


        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ─────────────────────────────────────────────────────────────
# BLEU Evaluation Route
# ─────────────────────────────────────────────────────────────

@app.route("/evaluate", methods=["GET"])
def evaluate():

    try:

        with open(
            "references.json",
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        predicted = []
        references = []


        for item in data:

            source_text = item["source"]

            translated_text, _ = translate_text(
                source_text,
                "en",
                "fr"
            )

            predicted.append(
                translated_text
            )

            references.append(
                item["reference"]
            )


        bleu = corpus_bleu(
            predicted,
            references
        )


        return jsonify({
            "bleu_score": bleu.score,
            "total_samples": len(predicted),
            "status": "success"
        })


    except Exception as e:

        print("BLEU EVALUATION ERROR:", str(e))

        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500


# ─────────────────────────────────────────────────────────────
# Render Local Development Run
# Gunicorn will ignore this block
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Starting Flask development server...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
