"""
Insight generation service layer for emotional analytics.
Produces rule-based AI guidance based on historical emotion data.

Emotion labels (from fine-tuned RoBERTa model):
    joy_excitement | affection | anger_disgust | sadness_grief
    fear_nervousness | cognitive | neutral
"""

from collections import Counter
from django.utils import timezone


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Valence score: positive → +1, negative → -1, neutral → 0
VALENCE = {
    'joy_excitement':   1,
    'affection':        1,
    'cognitive':        0,
    'neutral':          0,
    'sadness_grief':   -1,
    'fear_nervousness':-1,
    'anger_disgust':   -1,
}

# Intensity score for Chart.js trend line (1–5 scale)
INTENSITY = {
    'joy_excitement':   5,
    'affection':        4,
    'cognitive':        3,
    'neutral':          3,
    'sadness_grief':    2,
    'fear_nervousness': 1,
    'anger_disgust':    1,
}

POSITIVE_EMOTIONS = {'joy_excitement', 'affection'}
NEGATIVE_EMOTIONS = {'sadness_grief', 'fear_nervousness', 'anger_disgust'}


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

def detect_trend(emotions_list):
    """
    Analyze trend direction from the last 7 emotional entries.

    Args:
        emotions_list: List of emotion strings (most recent first)

    Returns:
        str: "upward", "downward", or "stable"
    """
    if not emotions_list:
        return "stable"

    recent = emotions_list[:7]
    scored = [VALENCE.get(e, 0) for e in recent]
    avg = sum(scored) / len(scored)

    if avg >= 0.4:
        return "upward"
    elif avg <= -0.4:
        return "downward"
    return "stable"


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def calculate_volatility(emotions_list):
    """
    Measure emotional variability as ratio of emotion switches to total entries.

    Args:
        emotions_list: List of emotion strings (chronological order)

    Returns:
        float: Volatility score (0.0 to 1.0)
    """
    if len(emotions_list) <= 1:
        return 0.0

    switches = sum(
        1 for i in range(len(emotions_list) - 1)
        if emotions_list[i] != emotions_list[i + 1]
    )
    return round(switches / (len(emotions_list) - 1), 2)


# ---------------------------------------------------------------------------
# AI insight generation
# ---------------------------------------------------------------------------

def generate_ai_insight(emotional_data):
    """
    Generate structured AI insight based on rule-based logic.

    Args:
        emotional_data: Dict with keys:
            - dominant_emotion (str)   — model label e.g. 'sadness_grief'
            - positive_ratio   (float) — 0–100
            - volatility       (float) — 0–1
            - trend_direction  (str)   — 'upward' | 'downward' | 'stable'

    Returns:
        Dict: { "summary": str, "recommendation": str }
    """
    dominant       = emotional_data.get("dominant_emotion", "neutral")
    positive_ratio = emotional_data.get("positive_ratio", 0)
    volatility     = emotional_data.get("volatility", 0)
    trend          = emotional_data.get("trend_direction", "stable")

    # Rule 1: Fear / nervousness with high volatility
    if dominant == "fear_nervousness" and volatility > 0.4:
        return {
            "summary": "Recurring anxiety and nervousness signals detected in your emotional patterns. Your entries show heightened emotional variability paired with fear as a dominant theme.",
            "recommendation": "Introduce structured breathing reset exercises during identified peak anxiety periods. Consider implementing a daily grounding routine to interrupt anxiety cycles before they intensify."
        }

    # Rule 2: Downward trend
    if trend == "downward":
        return {
            "summary": "A decline in positive emotional affect has been detected over your recent entries. Your baseline mood stability is shifting downward.",
            "recommendation": "Establish consistent daily routines and maintain regular sleep cycles. Structured daily activities help stabilize emotional baseline. Consider increasing journaling frequency to track the root triggers of this decline."
        }

    # Rule 3: Strong positive baseline
    if positive_ratio > 65:
        return {
            "summary": "Your emotional baseline shows predominantly positive affect. Joy, excitement, and warmth are well-represented in your recent reflections.",
            "recommendation": "Maintain your current journaling frequency to reinforce emotional stability. Document what conditions support this positive baseline — this pattern recognition helps predict and sustain wellbeing."
        }

    # Rule 4: High volatility
    if volatility > 0.5:
        return {
            "summary": "High emotional variability detected. Your emotional states are shifting frequently across entries, indicating unstable baseline patterns.",
            "recommendation": "Prioritize sleep cycle regulation and establish consistent daily structure. Emotional volatility often correlates with disrupted sleep and unstructured routines."
        }

    # Rule 5: Sadness / grief dominance
    if dominant == "sadness_grief":
        return {
            "summary": "Your emotional profile shows sadness and grief as dominant states. This pattern suggests sustained low mood or loss-related cycles in your recent reflections.",
            "recommendation": "Engage in structured social connection and light physical activity. Isolation amplifies negative emotional patterns. Establish small daily activities that create emotional counterweights to sadness."
        }

    # Rule 6: Anger / disgust dominance
    if dominant == "anger_disgust":
        return {
            "summary": "Frustration and anger appear as recurring themes in your recent entries. This may reflect unresolved tension or ongoing stressors.",
            "recommendation": "Try expressive writing to externalize and process frustration before it accumulates. Physical activity and structured breaks during high-stress periods can help discharge tension constructively."
        }

    # Rule 7: Upward trend
    if trend == "upward":
        return {
            "summary": "Your emotional trajectory is trending positively. Recent entries reflect an improving mood baseline.",
            "recommendation": "Continue the habits and routines that are contributing to this upward shift. Logging what's going well reinforces positive patterns and builds resilience."
        }

    # Default: balanced / stable
    return {
        "summary": "Moderate emotional variability observed. Your emotional states show normal fluctuation within a balanced range, with no dominant negative patterns.",
        "recommendation": "Continue your structured self-reflection practice. Regular journaling maintains baseline emotional awareness and helps identify patterns before they accumulate."
    }


# ---------------------------------------------------------------------------
# Chart.js trend data
# ---------------------------------------------------------------------------

def prepare_trend_data(entries):
    """
    Prepare daily trend data for Chart.js visualization.

    Args:
        entries: QuerySet of EmotionResult objects (ordered chronologically)

    Returns:
        Dict: { "labels": [...], "scores": [...], "has_data": bool }
    """
    date_scores = {}

    for result in entries.order_by("entry__created_at"):
        local_dt = timezone.localtime(result.entry.created_at)
        date_key = local_dt.date()
        score    = INTENSITY.get(result.emotion, 3)

        date_scores.setdefault(date_key, []).append(score)

    labels = []
    scores = []

    for date_key in sorted(date_scores):
        labels.append(date_key.strftime("%b %d"))
        avg = sum(date_scores[date_key]) / len(date_scores[date_key])
        scores.append(round(avg, 1))

    return {
        "labels":   labels,
        "scores":   scores,
        "has_data": len(labels) >= 2,
    }