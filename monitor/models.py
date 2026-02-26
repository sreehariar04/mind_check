from django.conf import settings
from django.db import models


class JournalEntry(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='journal_entries'
    )
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'JournalEntry #{self.id} by {self.user.username}'


class EmotionResult(models.Model):
    # Updated to match the 7 grouped labels from the fine-tuned RoBERTa model
    EMOTION_JOY         = 'joy_excitement'
    EMOTION_AFFECTION   = 'affection'
    EMOTION_ANGER       = 'anger_disgust'
    EMOTION_SADNESS     = 'sadness_grief'
    EMOTION_FEAR        = 'fear_nervousness'
    EMOTION_COGNITIVE   = 'cognitive'
    EMOTION_NEUTRAL     = 'neutral'

    EMOTION_CHOICES = [
        (EMOTION_JOY,       'Joy & Excitement'),
        (EMOTION_AFFECTION, 'Affection'),
        (EMOTION_ANGER,     'Anger & Disgust'),
        (EMOTION_SADNESS,   'Sadness & Grief'),
        (EMOTION_FEAR,      'Fear & Nervousness'),
        (EMOTION_COGNITIVE, 'Cognitive (Curiosity/Surprise)'),
        (EMOTION_NEUTRAL,   'Neutral'),
    ]

    # Human-friendly display names for templates
    EMOTION_DISPLAY = {
        'joy_excitement':  'Joy & Excitement',
        'affection':       'Affection',
        'anger_disgust':   'Anger & Disgust',
        'sadness_grief':   'Sadness & Grief',
        'fear_nervousness':'Fear & Nervousness',
        'cognitive':       'Curiosity & Surprise',
        'neutral':         'Neutral',
    }

    entry = models.OneToOneField(
        JournalEntry,
        on_delete=models.CASCADE,
        related_name='emotion_result'
    )
    emotion = models.CharField(max_length=30, choices=EMOTION_CHOICES)
    confidence = models.FloatField(help_text='Prediction confidence between 0 and 1.')
    analyzed_at = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50, default='roberta-v1')

    class Meta:
        ordering = ['-analyzed_at']

    def confidence_percent(self):
        return round(self.confidence * 100, 2)

    def display_emotion(self):
        return self.EMOTION_DISPLAY.get(self.emotion, self.emotion.replace('_', ' ').title())

    def __str__(self):
        return f'{self.display_emotion()} ({self.confidence_percent()}%)'


class UserPreference(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='preference'
    )
    notifications_enabled = models.BooleanField(default=True)
    dark_mode_enabled = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'Preferences for {self.user.username}'