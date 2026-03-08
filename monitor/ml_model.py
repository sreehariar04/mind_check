import os
import re
import logging

import numpy as np
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MIND_CHECK_MODEL_DIR", os.path.join(BASE_DIR, "saved_roberta"))
LE_PATH   = os.environ.get("MIND_CHECK_LE_PATH",   os.path.join(BASE_DIR, "label_encoder.pkl"))

MAX_SEQ_LEN          = 128
CONFIDENCE_THRESHOLD = 0.35
TOP2_GAP_THRESHOLD   = 0.10

NEGATION_TRIGGERS = r"\b(not|never|no|zero|nothing|hardly|stopped|would not|is not|do not|does not|cannot|can not)\b"
POSITIVE_TRIGGERS = r"\b(happy|excited|love|amazing|wonderful|fantastic|good|joy|fine)\b"
POSITIVE_EMOTIONS = {'joy_excitement', 'affection'}

NEGATIVE_CONTEXT  = r"\b(ruin|ruined|destroy|destroyed|mess|wreck|wrecked|hate|awful|terrible|worst|disappoint|disappointed|pathetic|useless)\b"
SARCASM_MARKERS   = r"\b(always|never|manage to|somehow|every single|typical)\b"

DECLINING_PATTERN = r"\b(not sure anymore|not so sure|not okay|have not been|haven't been|used to be|not the same|falling apart|losing it)\b"

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
    text = re.sub(r"[^a-zA-Z0-9!?.', ]+", ' ', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'(\w)\1{2,}', r'\1\1', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _split_sentences(text: str) -> list:
    """Split on . ! ? while keeping the delimiter. Filters out short fragments."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip().split()) >= 3]
    return sentences if sentences else [text]


def _has_negated_positive(text: str) -> bool:
    for m in re.finditer(NEGATION_TRIGGERS, text):
        window = text[m.start(): m.end() + 25]
        if re.search(POSITIVE_TRIGGERS, window):
            return True
    return False


def _has_sarcastic_affection(text: str) -> bool:
    if not re.search(r'\blove\b', text):
        return False
    return bool(re.search(NEGATIVE_CONTEXT, text) or re.search(SARCASM_MARKERS, text))


def _has_declining_wellbeing(text: str) -> bool:
    return bool(re.search(DECLINING_PATTERN, text))


_tokenizer = None
_model     = None
_le        = None
_device    = None


def _load_model():
    global _tokenizer, _model, _le, _device

    if _model is not None:
        return

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

    logger.info("mind_check: RoBERTa model loaded on %s -- classes: %s",
                _device, list(_le.classes_))


def _infer_probs(text: str) -> np.ndarray:
    """Run TTA inference on a single cleaned text. Returns averaged probs."""
    variants  = [text, text.lower(), text.rstrip('.') + '.']
    all_probs = []
    for variant in variants:
        inputs = _tokenizer(
            variant,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = torch.softmax(_model(**inputs).logits, dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)


def _apply_rules(probs: np.ndarray, cleaned: str):
    """Apply post-inference suppression rules. Returns (adjusted_probs, fired_flag)."""
    fired = False

    if _has_negated_positive(cleaned):
        for cls_name in POSITIVE_EMOTIONS:
            if cls_name in _le.classes_:
                probs[list(_le.classes_).index(cls_name)] *= 0.4
        probs = probs / probs.sum()
        fired = True
        logger.info("mind_check: negation suppression for: '%s'", cleaned[:60])

    if _has_sarcastic_affection(cleaned):
        if 'affection' in _le.classes_:
            probs[list(_le.classes_).index('affection')] *= 0.2
        probs = probs / probs.sum()
        fired = True
        logger.info("mind_check: sarcastic affection suppression for: '%s'", cleaned[:60])

    if _has_declining_wellbeing(cleaned):
        if 'cognitive' in _le.classes_:
            probs[list(_le.classes_).index('cognitive')] *= 0.3
        if 'sadness_grief' in _le.classes_:
            probs[list(_le.classes_).index('sadness_grief')] *= 1.5
        probs = probs / probs.sum()
        logger.info("mind_check: declining wellbeing for: '%s'", cleaned[:60])

    return probs, fired


def predict_emotion(text: str) -> dict:
    _load_model()

    cleaned_full = _clean_text(text)
    if not cleaned_full:
        return {"emotion": "neutral", "confidence": 1.0, "top3": {"neutral": 1.0}}

    sentences = _split_sentences(text)

    if len(sentences) <= 1:
        # Short single-sentence input
        probs, negation_fired = _apply_rules(_infer_probs(cleaned_full), cleaned_full)

    else:
        # Multi-sentence journal entry:
        # Classify each sentence individually, weight the last sentence 2x
        # because journal entries tend to end on the dominant emotional state
        n           = len(sentences)
        weights     = [1.0] * n
        weights[-1] = 2.0
        total_w     = sum(weights)

        combined       = np.zeros(len(_le.classes_))
        negation_fired = False

        for i, sent in enumerate(sentences):
            cleaned_sent = _clean_text(sent)
            if not cleaned_sent:
                continue
            probs_sent, fired = _apply_rules(_infer_probs(cleaned_sent), cleaned_sent)
            combined += probs_sent * (weights[i] / total_w)
            if fired:
                negation_fired = True

        # Also apply rules on full text to catch cross-sentence patterns
        # e.g. "putting on a smile... sit in silence"
        _, full_fired = _apply_rules(_infer_probs(cleaned_full), cleaned_full)
        if full_fired:
            negation_fired = True

        probs = combined / combined.sum()

    pred_idx        = int(np.argmax(probs))
    all_pairs       = sorted(zip(_le.classes_, probs), key=lambda x: -x[1])
    top3_dict       = {k: round(float(v), 4) for k, v in all_pairs[:3]}
    predicted_label = _le.inverse_transform([pred_idx])[0]
    confidence      = round(float(probs[pred_idx]), 4)

    top2_gap = all_pairs[0][1] - all_pairs[1][1]
    if confidence < CONFIDENCE_THRESHOLD or top2_gap < TOP2_GAP_THRESHOLD:
        predicted_label = "neutral"

    result = {
        "emotion":    predicted_label,
        "confidence": confidence,
        "top3":       top3_dict,
    }
    if negation_fired:
        result["negation_override"] = True

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = [
        ("I've been putting on a smile at work but when I get home I just sit in silence.", "sadness_grief"),
        ("Today was great on paper. Good weather, good food, good company. I still felt off though.", "sadness_grief"),
        ("Went to the store. Made dinner. Watched something. Went to bed.", "neutral"),
        ("I snapped at someone I care about today. I feel terrible but also I was right.", "sadness_grief"),
        ("Had a really good day today, got a lot done and actually felt proud of myself for once.", "joy_excitement"),
        ("Couldn't get out of bed until noon. Not because I was tired, just didn't see the point.", "sadness_grief"),
        ("I just got promoted, I can't believe it!", "joy_excitement"),
        ("I love how you always manage to ruin everything.", "anger_disgust"),
        ("I thought I was okay but I'm not sure anymore.", "sadness_grief"),
        ("Not bad at all, actually really enjoyed it.", "joy_excitement"),
    ]
    print(f"\n{'Input':<70}  {'Expected':<30}  {'Got':<25}  Conf")
    print("-" * 145)
    for text, expected in samples:
        r = predict_emotion(text)
        print(f"{text:<70}  {expected:<30}  {r['emotion']:<25}  {r['confidence']:.1%}")