"""
monitor/ml_model.py
-------------------
Loads the fine-tuned RoBERTa model (saved_roberta/) and label encoder
(label_encoder.pkl) produced by the Colab training notebook, then
exposes a single public function:

    predict_emotion(text: str) -> dict

Return format (matches what the existing Django views already expect):
    {
        "emotion":    str,   # e.g. "joy_excitement"
        "confidence": float, # 0.0 – 1.0, rounded to 4 d.p.
        "top3":       dict,  # {"emotion_name": score, ...} top-3 predictions
    }

Model files expected at (relative to manage.py / project root):
    ./saved_roberta/          ← saved_pretrained() output folder
    ./label_encoder.pkl       ← joblib.dump(le, ...)

To change the path, set the env vars:
    MIND_CHECK_MODEL_DIR   (default: saved_roberta)
    MIND_CHECK_LE_PATH     (default: label_encoder.pkl)
"""

import os
import re
import logging

import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (override via environment variables)
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.environ.get("MIND_CHECK_MODEL_DIR",
                            os.path.join(BASE_DIR, "saved_roberta"))
LE_PATH    = os.environ.get("MIND_CHECK_LE_PATH",
                            os.path.join(BASE_DIR, "label_encoder.pkl"))

MAX_SEQ_LEN = 128

# If the model's top prediction is below this confidence, fall back to neutral.
# Prevents overconfident mislabelling of mundane / ambiguous entries.
CONFIDENCE_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Text cleaning  (mirrors the training notebook exactly)
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "won't":"will not","can't":"cannot","don't":"do not",
    "doesn't":"does not","didn't":"did not","isn't":"is not",
    "aren't":"are not","wasn't":"was not","weren't":"were not",
    "haven't":"have not","hasn't":"has not","hadn't":"had not",
    "wouldn't":"would not","shouldn't":"should not","couldn't":"could not",
    "i'm":"i am","i've":"i have","i'll":"i will","i'd":"i would",
    "you're":"you are","you've":"you have","you'll":"you will",
    "he's":"he is","she's":"she is","it's":"it is",
    "we're":"we are","we've":"we have","they're":"they are",
    "they've":"they have","that's":"that is","there's":"there is",
    "what's":"what is","let's":"let us",
}

def _clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'r/\w+|u/\w+', '', text)
    for c, e in CONTRACTIONS.items():
        text = text.replace(c, e)
    text = re.sub(r'[^\w\s!?]', ' ', text)
    text = re.sub(r'!{2,}', '!!', text)
    text = re.sub(r'\?{2,}', '??', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# Lazy model loading  (loaded once on first call, not at import time)
# ---------------------------------------------------------------------------
_tokenizer = None
_model     = None
_le        = None
_device    = None

def _load_model():
    """Load tokenizer, model and label-encoder into module-level globals."""
    global _tokenizer, _model, _le, _device

    if _model is not None:
        return  # already loaded

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Model directory not found: {MODEL_DIR}\n"
            "Run the training notebook and copy saved_roberta/ next to manage.py."
        )
    if not os.path.isfile(LE_PATH):
        raise FileNotFoundError(
            f"Label encoder not found: {LE_PATH}\n"
            "Run the training notebook and copy label_encoder.pkl next to manage.py."
        )

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _model.to(_device)
    _model.eval()
    _le        = joblib.load(LE_PATH)

    logger.info("mind_check: RoBERTa model loaded on %s — classes: %s",
                _device, list(_le.classes_))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_emotion(text: str) -> dict:
    """
    Predict the emotion group for a journal entry.

    Parameters
    ----------
    text : str
        Raw user-submitted journal text.

    Returns
    -------
    dict with keys:
        emotion    (str)  — predicted emotion group label
        confidence (float)— probability of the top prediction (0–1)
        top3       (dict) — top-3 {label: probability} pairs
    """
    _load_model()

    cleaned = _clean_text(text)
    if not cleaned:
        return {"emotion": "neutral", "confidence": 1.0,
                "top3": {"neutral": 1.0}}

    inputs = _tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    probs      = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = round(float(probs[pred_idx]), 4)

    top3_pairs = sorted(
        zip(_le.classes_, probs),
        key=lambda x: -x[1]
    )[:3]

    # Fall back to neutral when the model is not confident enough.
    predicted_label = _le.inverse_transform([pred_idx])[0]
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = "neutral"

    return {
        "emotion":    predicted_label,
        "confidence": confidence,
        "top3":       {k: round(float(v), 4) for k, v in top3_pairs},
    }


# ---------------------------------------------------------------------------
# Quick smoke-test  (python monitor/ml_model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        "I just got promoted, I can't believe it!",
        "Everything feels so pointless lately.",
        "Why would anyone do something like that??",
        "I feel so scared and nervous about tomorrow.",
        "I love how caring and kind you are.",
    ]
    for s in samples:
        r = predict_emotion(s)
        print(f"Text      : {s!r}")
        print(f"Emotion   : {r['emotion']}  ({r['confidence']:.1%})")
        print(f"Top 3     : {r['top3']}\n")