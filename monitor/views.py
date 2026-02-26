import calendar
import csv
from urllib.parse import urlencode
from collections import defaultdict
from datetime import timedelta, datetime

from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.db.models import Count
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.utils.dateparse import parse_date

from .forms import (
    EmailAuthenticationForm,
    JournalEntryForm,
    ProfileForm,
    RegisterForm,
    UserPreferenceForm,
)
from .ml_model import predict_emotion
from .models import EmotionResult, JournalEntry, UserPreference


# ---------------------------------------------------------------------------
# Emotion label sets (use model constants, not raw strings)
# ---------------------------------------------------------------------------
POSITIVE_EMOTIONS = {EmotionResult.EMOTION_JOY, EmotionResult.EMOTION_AFFECTION}
NEGATIVE_EMOTIONS = {EmotionResult.EMOTION_ANGER, EmotionResult.EMOTION_SADNESS, EmotionResult.EMOTION_FEAR}

# Valence scores for trend calculation
VALENCE_SCORES = {
    EmotionResult.EMOTION_JOY:       2,
    EmotionResult.EMOTION_AFFECTION: 1,
    EmotionResult.EMOTION_COGNITIVE: 0,
    EmotionResult.EMOTION_NEUTRAL:   0,
    EmotionResult.EMOTION_SADNESS:  -1,
    EmotionResult.EMOTION_FEAR:     -2,
    EmotionResult.EMOTION_ANGER:    -2,
}


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------

def landing_page(request):
    return render(request, 'monitor/landing.html')


def register_page(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully.')
            return redirect('dashboard')
    else:
        form = RegisterForm()

    return render(request, 'monitor/register.html', {'form': form})


def login_page(request):
    return LoginView.as_view(
        template_name='registration/login.html',
        authentication_form=EmailAuthenticationForm,
    )(request)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@login_required
def dashboard_page(request):
    results = EmotionResult.objects.filter(entry__user=request.user).select_related('entry')
    entries = JournalEntry.objects.filter(user=request.user).select_related('emotion_result')

    total_entries  = entries.count()
    positive_count = results.filter(emotion__in=POSITIVE_EMOTIONS).count()
    positive_ratio = round((positive_count / total_entries) * 100, 1) if total_entries else 0
    streak         = _calculate_streak(entries)

    calendar_weeks, month_name = _build_month_calendar(entries)
    month, year = month_name.split(' ')

    emotion_counts = _emotion_counts(results)

    context = {
        'total_entries':  total_entries,
        'positive_ratio': positive_ratio,
        'streak':         streak,
        'month_days':     calendar_weeks,
        'month':          month,
        'year':           year,
        'emotion_counts': emotion_counts,
    }
    return render(request, 'monitor/dashboard.html', context)


# ---------------------------------------------------------------------------
# Journal submission
# ---------------------------------------------------------------------------

@login_required
def journal_page(request):
    if request.method == 'POST':
        form = JournalEntryForm(request.POST)
        if form.is_valid():
            entry = form.save(commit=False)
            entry.user = request.user
            entry.save()

            # predict_emotion() returns a dict: {emotion, confidence, top3}
            prediction = predict_emotion(entry.content)
            result = EmotionResult.objects.create(
                entry=entry,
                emotion=prediction['emotion'],
                confidence=prediction['confidence'],
                model_version='roberta-v1',
            )

            messages.success(request, 'Journal analyzed successfully.')
            return redirect('emotional_results', entry_id=result.entry_id)
    else:
        form = JournalEntryForm()

    return render(request, 'monitor/journal.html', {'form': form})


# ---------------------------------------------------------------------------
# Emotional results
# ---------------------------------------------------------------------------

@login_required
def emotional_results_page(request, entry_id):
    entry  = get_object_or_404(JournalEntry.objects.select_related('emotion_result'), id=entry_id, user=request.user)
    result = get_object_or_404(EmotionResult, entry=entry)

    categories = [emotion for emotion, _ in EmotionResult.EMOTION_CHOICES]
    radar_data = {emotion: 12 for emotion in categories}
    radar_data[result.emotion] = max(25, int(result.confidence * 100))

    suggestions = {
        EmotionResult.EMOTION_JOY: (
            'Your reflection radiates positive energy and excitement. '
            'Maintain this momentum with consistent routines and gratitude-based journaling.'
        ),
        EmotionResult.EMOTION_AFFECTION: (
            'Your writing reflects warmth and connection. '
            'Nurturing relationships and expressing appreciation amplifies this positive state.'
        ),
        EmotionResult.EMOTION_SADNESS: (
            'Your writing suggests low emotional energy. '
            'Gentle routines, social connection, and structured sleep may help regulate mood.'
        ),
        EmotionResult.EMOTION_FEAR: (
            'The current profile reflects elevated fear and nervousness. '
            'Focus on grounding techniques and short certainty-based planning to reduce mental load.'
        ),
        EmotionResult.EMOTION_ANGER: (
            'Your reflection shows heightened frustration markers. '
            'Pause before major decisions and use brief breathing breaks to de-intensify reactions.'
        ),
        EmotionResult.EMOTION_COGNITIVE: (
            'Your entries show active curiosity and mental engagement. '
            'Channel this reflective energy into structured problem-solving or creative outlets.'
        ),
        EmotionResult.EMOTION_NEUTRAL: (
            'Your emotional tone is currently balanced. '
            'Continue consistent journaling to monitor subtle shifts and maintain mental stability.'
        ),
    }

    context = {
        'entry':        entry,
        'result':       result,
        'radar_labels': categories,
        'radar_values': [radar_data[label] for label in categories],
        'suggestion':   suggestions.get(result.emotion, suggestions[EmotionResult.EMOTION_NEUTRAL]),
    }
    return render(request, 'monitor/emotional_results.html', context)


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

@login_required
def insights_page(request):
    from .services import detect_trend, calculate_volatility, generate_ai_insight, prepare_trend_data

    user_results = (
        EmotionResult.objects
        .filter(entry__user=request.user)
        .select_related('entry')
        .order_by('-entry__created_at')
    )

    emotions_chronological = list(user_results.order_by('entry__created_at').values_list('emotion', flat=True))
    emotions_recent_first  = list(user_results.values_list('emotion', flat=True))

    total_entries  = user_results.count()
    emotion_counts = _emotion_counts(user_results)

    dominant_emotion = emotion_counts[0]['emotion'] if emotion_counts else EmotionResult.EMOTION_NEUTRAL

    positive_count = sum(1 for e in emotions_recent_first if e in POSITIVE_EMOTIONS)
    positive_ratio = round((positive_count / total_entries) * 100, 1) if total_entries else 0

    trend_direction = detect_trend(emotions_recent_first)
    volatility      = calculate_volatility(emotions_chronological)

    ai_insight = generate_ai_insight({
        'dominant_emotion': dominant_emotion,
        'positive_ratio':   positive_ratio,
        'volatility':       volatility,
        'trend_direction':  trend_direction,
    })

    trend_data       = prepare_trend_data(user_results)
    weekly_summary   = _weekly_summary(user_results)
    monthly_summary  = _monthly_summary(user_results)
    pattern_insights = _pattern_insights(user_results, weekly_summary['results'])

    context = {
        'total_entries':    total_entries,
        'emotion_counts':   emotion_counts,
        'dominant_emotion': dominant_emotion,
        'positive_ratio':   positive_ratio,
        'trend_direction':  trend_direction,
        'volatility':       volatility,
        'pattern_summary':  ai_insight['summary'],
        'ai_tip':           ai_insight['recommendation'],
        'trend_labels':     trend_data['labels'],
        'trend_scores':     trend_data['scores'],
        'trend_has_data':   trend_data['has_data'],
        'weekly_summary':   weekly_summary,
        'monthly_summary':  monthly_summary,
        'pattern_insights': pattern_insights,
    }
    return render(request, 'monitor/insights.html', context)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@login_required
def history_page(request):
    query      = request.GET.get('q', '').strip()
    emotion    = request.GET.get('emotion', '').strip()
    date_value = request.GET.get('date', '').strip()
    parsed_date = parse_date(date_value) if date_value else None

    entries = JournalEntry.objects.filter(user=request.user).select_related('emotion_result')

    if query:
        entries = entries.filter(content__icontains=query)
    if emotion:
        entries = entries.filter(emotion_result__emotion=emotion)
    if parsed_date:
        entries = entries.filter(created_at__date=parsed_date)

    if request.method == 'POST':
        entry_id = request.POST.get('entry_id')
        if entry_id:
            JournalEntry.objects.filter(id=entry_id, user=request.user).delete()
            messages.success(request, 'Entry deleted successfully.')
            params = {k: v for k, v in [('q', query), ('emotion', emotion), ('date', date_value)] if v}
            return redirect(f"{request.path}?{urlencode(params)}" if params else request.path)

    context = {
        'entries':        entries,
        'query':          query,
        'emotion':        emotion,
        'date':           date_value,
        'emotion_choices': EmotionResult.EMOTION_CHOICES,
    }
    return render(request, 'monitor/history.html', context)


@login_required
def history_by_date(request, selected_date):
    try:
        parsed_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
        entries = (
            JournalEntry.objects
            .filter(user=request.user, created_at__date=parsed_date)
            .select_related('emotion_result')
            .order_by('-created_at')
        )

        if request.method == 'POST':
            entry_id = request.POST.get('entry_id')
            if entry_id:
                JournalEntry.objects.filter(id=entry_id, user=request.user).delete()
                messages.success(request, 'Entry deleted successfully.')
                return redirect('history_by_date', selected_date=selected_date)

        context = {
            'entries':        entries,
            'selected_date':  parsed_date,
            'emotion_choices': EmotionResult.EMOTION_CHOICES,
        }
        return render(request, 'monitor/history.html', context)

    except (ValueError, TypeError):
        return redirect('history')


# ---------------------------------------------------------------------------
# Settings / Profile
# ---------------------------------------------------------------------------

@login_required
def settings_page(request):
    preference, _ = UserPreference.objects.get_or_create(user=request.user)

    if request.method == 'POST':
        action = request.POST.get('action', 'save')

        if action == 'export':
            return _export_entries_csv(request.user)

        if action == 'delete_account':
            user = request.user
            logout(request)
            user.delete()
            messages.success(request, 'Your account has been deleted.')
            return redirect('landing')

        profile_form    = ProfileForm(request.POST, instance=request.user)
        preference_form = UserPreferenceForm(request.POST, instance=preference)
        if profile_form.is_valid() and preference_form.is_valid():
            profile_form.save()
            preference_form.save()
            messages.success(request, 'Settings updated successfully.')
            return redirect('settings')
    else:
        profile_form    = ProfileForm(instance=request.user)
        preference_form = UserPreferenceForm(instance=preference)

    context = {
        'profile_form':    profile_form,
        'preference_form': preference_form,
    }
    return render(request, 'monitor/settings.html', context)


@login_required
def profile_page(request):
    return redirect('settings')


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@login_required
@require_POST
def predict_emotion_api(request):
    form = JournalEntryForm(request.POST)
    if not form.is_valid():
        return JsonResponse({'error': 'Invalid input.'}, status=400)

    entry = form.save(commit=False)
    entry.user = request.user
    entry.save()

    prediction = predict_emotion(entry.content)
    result = EmotionResult.objects.create(
        entry=entry,
        emotion=prediction['emotion'],
        confidence=prediction['confidence'],
        model_version='roberta-v1',
    )

    return JsonResponse({
        'emotion':    result.emotion,
        'display':    result.display_emotion(),
        'confidence': result.confidence_percent(),
        'message':    _supportive_message(result.emotion),
    })


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _calculate_streak(entries_queryset):
    dates = sorted({entry.created_at.date() for entry in entries_queryset}, reverse=True)
    if not dates:
        return 0
    today = timezone.localdate()
    start = today if dates[0] == today else today - timedelta(days=1)
    if dates[0] < start:
        return 0
    streak, cursor, date_set = 0, start, set(dates)
    while cursor in date_set:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _calculate_volatility_index(results_queryset):
    sequence = list(results_queryset.order_by('entry__created_at').values_list('emotion', flat=True))
    if len(sequence) < 2:
        return 0
    transitions = sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])
    return round((transitions / (len(sequence) - 1)) * 100, 1)


def _emotion_counts(results_queryset):
    total = results_queryset.count()
    rows  = results_queryset.values('emotion').annotate(total=Count('id')).order_by('-total')
    return [
        {
            'emotion': row['emotion'],
            'count':   row['total'],
            'percent': round((row['total'] / total) * 100, 1) if total else 0,
        }
        for row in rows
    ]


def _weekly_summary(results_queryset):
    start   = timezone.now() - timedelta(days=7)
    results = results_queryset.filter(entry__created_at__gte=start)
    top     = results.values('emotion').annotate(total=Count('id')).order_by('-total').first()
    return {'count': results.count(), 'top': top, 'results': results}


def _monthly_summary(results_queryset):
    start   = timezone.now() - timedelta(days=30)
    results = results_queryset.filter(entry__created_at__gte=start)
    top     = results.values('emotion').annotate(total=Count('id')).order_by('-total').first()
    return {'count': results.count(), 'top': top, 'results': results}


def _weekly_trend_scores(results_queryset):
    start  = timezone.now().date() - timedelta(days=6)
    labels, scores = [], []
    for day_index in range(7):
        date_value = start + timedelta(days=day_index)
        labels.append(date_value.strftime('%b %d'))
        day_results = results_queryset.filter(entry__created_at__date=date_value)
        if not day_results.exists():
            scores.append(0)
            continue
        day_scores = [VALENCE_SCORES.get(r.emotion, 0) for r in day_results]
        scores.append(round(sum(day_scores) / len(day_scores), 2))
    return labels, scores


def _pattern_insights(all_results_queryset, weekly_results_queryset):
    this_week       = weekly_results_queryset
    last_week_start = timezone.now() - timedelta(days=14)
    last_week_end   = timezone.now() - timedelta(days=7)
    last_week       = all_results_queryset.filter(
        entry__created_at__gte=last_week_start,
        entry__created_at__lt=last_week_end,
    )

    insights = []

    for emotion in NEGATIVE_EMOTIONS:
        count = this_week.filter(emotion=emotion).count()
        if count >= 2:
            label = EmotionResult.EMOTION_DISPLAY.get(emotion, emotion)
            insights.append(f"You experienced {label.lower()} {count} times this week.")

    this_neg  = this_week.filter(emotion__in=NEGATIVE_EMOTIONS).count()
    last_neg  = last_week.filter(emotion__in=NEGATIVE_EMOTIONS).count()
    this_pos  = this_week.filter(emotion__in=POSITIVE_EMOTIONS).count()
    last_pos  = last_week.filter(emotion__in=POSITIVE_EMOTIONS).count()

    if this_neg < last_neg:
        insights.append('Negative emotions have reduced compared to last week.')
    elif this_neg > last_neg:
        insights.append('Negative emotions were higher than last week.')

    if this_pos > last_pos:
        insights.append('Your positive emotions increased compared to last week.')

    return insights


def _supportive_message(emotion):
    messages_map = {
        EmotionResult.EMOTION_JOY:       'You seem energised and positive today. Keep nurturing what is working.',
        EmotionResult.EMOTION_AFFECTION:  'Warmth and connection come through in your words. That is worth holding onto.',
        EmotionResult.EMOTION_SADNESS:    'Your tone feels a bit low. Gentle routines and connection can help.',
        EmotionResult.EMOTION_FEAR:       'Some fear signals appear. Grounding exercises may help you feel safer.',
        EmotionResult.EMOTION_ANGER:      'There are signs of frustration. A brief reset can ease intensity.',
        EmotionResult.EMOTION_COGNITIVE:  'Your mind seems active and curious today. Channel that energy constructively.',
        EmotionResult.EMOTION_NEUTRAL:    'Your emotional tone looks balanced. Keep journaling to maintain clarity.',
    }
    return messages_map.get(emotion, messages_map[EmotionResult.EMOTION_NEUTRAL])


def _build_month_calendar(entries_queryset):
    today  = timezone.localdate()
    year, month = today.year, today.month

    month_entries = entries_queryset.filter(
        created_at__year=year, created_at__month=month
    ).order_by('-created_at')

    emotion_by_date = {}
    for entry in month_entries:
        date_key = timezone.localtime(entry.created_at).date()
        if date_key not in emotion_by_date:
            emotion_by_date[date_key] = getattr(entry.emotion_result, 'emotion', None)

    calendar_weeks = []
    for week in calendar.Calendar(firstweekday=0).monthdatescalendar(year, month):
        calendar_weeks.append([
            {
                'day':       day,
                'day_number': day.day,
                'in_month':  day.month == month,
                'is_today':  day == today,
                'emotion':   emotion_by_date.get(day),
                'date_str':  day.strftime('%Y-%m-%d'),
            }
            for day in week
        ])

    return calendar_weeks, today.strftime('%B %Y')


def _build_emotion_distribution(results_queryset):
    total     = results_queryset.count()
    counts_qs = results_queryset.values('emotion').annotate(total=Count('id'))
    counts    = {row['emotion']: row['total'] for row in counts_qs}
    return [
        {
            'emotion': emotion,
            'count':   counts.get(emotion, 0),
            'percent': round((counts.get(emotion, 0) / total) * 100, 1) if total else 0,
        }
        for emotion, _ in EmotionResult.EMOTION_CHOICES
    ]


def _export_entries_csv(user):
    entries  = JournalEntry.objects.filter(user=user).select_related('emotion_result').order_by('-created_at')
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="mindcheck_entries.csv"'

    writer = csv.writer(response)
    writer.writerow(['Date', 'Journal Text', 'Emotion', 'Confidence'])
    for entry in entries:
        er = getattr(entry, 'emotion_result', None)
        writer.writerow([
            entry.created_at.strftime('%Y-%m-%d %H:%M'),
            entry.content,
            er.display_emotion() if er else 'N/A',
            er.confidence if er else 'N/A',
        ])
    return response