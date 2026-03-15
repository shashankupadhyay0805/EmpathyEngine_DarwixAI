"""
Emotion detection for text using HuggingFace Transformers with fallback to VADER/TextBlob.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from app.config import EMOTION_LABELS, EMOTION_MODEL_HF, EMOTION_TO_CATEGORY
from app.utils import get_logger

logger = get_logger(__name__)

EmotionCategory = Literal["happy", "frustrated", "neutral"]


@dataclass
class EmotionResult:
    """Result of emotion detection."""

    emotion: EmotionCategory
    raw_label: str
    confidence: float

    def __str__(self) -> str:
        return f"{self.emotion} (raw: {self.raw_label}, conf: {self.confidence:.2f})"


def _detect_with_transformers(text: str) -> Optional[EmotionResult]:
    """Use HuggingFace pipeline for emotion classification."""
    try:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model=EMOTION_MODEL_HF,
            top_k=1,
            truncation=True,
            max_length=512,
        )
        result = classifier(text[:512])[0]
        label = result["label"].lower().replace(" ", "_")
        score = float(result["score"])
        category = EMOTION_TO_CATEGORY.get(label, "neutral")
        return EmotionResult(emotion=category, raw_label=label, confidence=score)
    except Exception as e:
        logger.warning("HuggingFace emotion detection failed: %s", e)
        return None


def _detect_with_vader(text: str) -> EmotionResult:
    """Fallback: use VADER for sentiment, map to emotion category."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            return EmotionResult(
                emotion="happy",
                raw_label="positive",
                confidence=min(1.0, (compound + 1) / 2),
            )
        if compound <= -0.05:
            return EmotionResult(
                emotion="frustrated",
                raw_label="negative",
                confidence=min(1.0, (1 - compound) / 2),
            )
        return EmotionResult(
            emotion="neutral",
            raw_label="neutral",
            confidence=1.0 - abs(compound),
        )
    except ImportError:
        return _detect_with_textblob(text)


def _detect_with_textblob(text: str) -> EmotionResult:
    """Fallback: use TextBlob polarity for simple sentiment."""
    try:
        from textblob import TextBlob

        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        if polarity > 0.1:
            return EmotionResult(
                emotion="happy",
                raw_label="positive",
                confidence=min(1.0, (polarity + 1) / 2),
            )
        if polarity < -0.1:
            return EmotionResult(
                emotion="frustrated",
                raw_label="negative",
                confidence=min(1.0, (1 - polarity) / 2),
            )
        return EmotionResult(
            emotion="neutral",
            raw_label="neutral",
            confidence=1.0 - abs(polarity),
        )
    except ImportError:
        logger.warning("No emotion backend available; defaulting to neutral.")
        return EmotionResult(emotion="neutral", raw_label="neutral", confidence=0.5)


def detect_emotion(text: str, use_fallback_only: bool = False) -> EmotionResult:
    """
    Detect the dominant emotion in the given text.

    Uses HuggingFace Transformers by default; falls back to VADER, then TextBlob
    if Transformers is unavailable or fails.

    Args:
        text: Input text to analyze.
        use_fallback_only: If True, skip HuggingFace and use VADER/TextBlob only.

    Returns:
        EmotionResult with emotion category, raw label, and confidence.
    """
    if not text or not text.strip():
        return EmotionResult(emotion="neutral", raw_label="neutral", confidence=0.0)

    if not use_fallback_only:
        result = _detect_with_transformers(text.strip())
        if result is not None:
            logger.debug("Emotion (transformers): %s", result)
            return result

    result = _detect_with_vader(text.strip())
    logger.debug("Emotion (fallback): %s", result)
    return result
