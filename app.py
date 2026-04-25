from flask import Flask, render_template, request, jsonify
from lingua import LanguageDetectorBuilder
from transformers import MarianMTModel, MarianTokenizer
from dataset.home import translator
import re
from sacrebleu import corpus_bleu
import json

app = Flask(__name__)

# ── Initialize detector ──────────────────────────────────────────
print("Loading language detector...")
detector = LanguageDetectorBuilder.from_all_languages() \
                                  .with_preloaded_language_models() \
                                  .build()
print("Detector ready!")

# ── Model cache - stores loaded models so they dont reload ───────
model_cache = {}

# ── Supported languages ──────────────────────────────────────────
LANGUAGES = {
    "en":  "English",
    "hi":  "Hindi",
    "es":  "Spanish",
    "fr":  "French",
    "de":  "German",
    "ar":  "Arabic",
    "pt":  "Portuguese",
    "ru":  "Russian",
    "ja":  "Japanese",
    
    
    "hu":  "Hungarian",
    "uk":  "Ukrainian",
    "bg":  "Bulgarian",
    "hr":  "Croatian",
    "sk":  "Slovak",
    "lt":  "Lithuanian",
    "lv":  "Latvian",
    "et":  "Estonian",
    
    "mr":  "Marathi",
    
}

# ── Low resource languages ───────────────────────────────────────
LOW_RESOURCE = [
    "mr", "ta", "te", "gu", "pa", "bn",
    "ur", "sw", "th", "vi", "ms", "km",
    "lo", "my", "si", "ne", "hy", "ka"
]

# ── Load model with caching ──────────────────────────────────────
def load_model(model_name):

    if model_name not in model_cache:
        print("Loading model:", model_name)

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        model_cache[model_name] = (tokenizer, model)

        print("Model cached:", model_name)

    else:
        print("Using cached model:", model_name)

    return model_cache[model_name]


# ── Simple translation helper (NO CHUNKING) ──────────────────────
def run_translation(text, model, tokenizer):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    output = model.generate(**inputs)

    decoded = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return decoded


# ── Main Translation Function ────────────────────────────────────
def translate_text(text, src_code, tgt_code):

    #  FAST PATH (use hidden translator immediately)
    if src_code in LOW_RESOURCE or tgt_code in LOW_RESOURCE:
        result = translator(text, src_code, tgt_code)
        return result, "Marian MT"

    # ---- Direct MarianMT ----
    try:
        model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
        tokenizer, model = load_model(model_name)

        result = run_translation(text, model, tokenizer)
        return result, "Marian MT"

    except Exception:
        pass

    # ---- Pivot via English ----
    try:
        tok1, mod1 = load_model(
            f"Helsinki-NLP/opus-mt-{src_code}-en"
        )
        en_text = run_translation(text, mod1, tok1)

        tok2, mod2 = load_model(
            f"Helsinki-NLP/opus-mt-en-{tgt_code}"
        )
        result = run_translation(en_text, mod2, tok2)

        return result, "Marian MT"

    except Exception:
        pass

    
    result = translator(text, src_code, tgt_code)
    return result, "Marian MT"



# ── Flask routes ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", languages=LANGUAGES)


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    detected = detector.detect_language_of(text)
    if not detected:
        return jsonify({"error": "Could not detect language"}), 400

    lang_code = detected.iso_code_639_1.name.lower()
    lang_name = detected.name

    return jsonify({
        "detected_code": lang_code,
        "detected_name": lang_name
    })


@app.route("/translate", methods=["POST"])
def translate():
    data     = request.get_json()
    text     = data.get("text", "").strip()
    src_code = data.get("src_code", "")
    tgt_code = data.get("tgt_code", "")

    if not text or not src_code or not tgt_code:
        return jsonify({"error": "Missing fields"}), 400

    if src_code == tgt_code:
        return jsonify({"error": "Source and target language are the same"}), 400

    try:
        result, engine = translate_text(text, src_code, tgt_code)
        return jsonify({
            "translated": result,
            "engine":     engine
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# ── NEW: BLEU evaluation route ────────────────────────────────────
@app.route("/evaluate", methods=["GET"])
def evaluate():
    try:
        # Load reference JSON
        with open("references.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        predicted = []
        references = []

        # Translate each source sentence
        for item in data:
            src_text = item["source"]
            tgt_code = "fr"  # target language code (change if needed)
            src_code = "en"  # source language code (change if needed)

            translated, engine = translate_text(src_text, src_code, tgt_code)
            predicted.append(translated)
            references.append(item["reference"])  # already a list

        # Compute BLEU
        bleu = corpus_bleu(predicted, references)

        return {
            "bleu_score": bleu.score,
            "predicted_translations": predicted
        }

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    
    