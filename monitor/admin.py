from django.contrib import admin

from .models import EmotionResult, JournalEntry, UserPreference


@admin.register(JournalEntry)
class JournalEntryAdmin(admin.ModelAdmin):
	list_display = ('id', 'user', 'created_at')
	search_fields = ('user__username', 'content')
	list_filter = ('created_at',)


@admin.register(EmotionResult)
class EmotionResultAdmin(admin.ModelAdmin):
	list_display = ('id', 'entry', 'emotion', 'confidence', 'analyzed_at')
	list_filter = ('emotion', 'analyzed_at')


@admin.register(UserPreference)
class UserPreferenceAdmin(admin.ModelAdmin):
	list_display = ('user', 'notifications_enabled', 'dark_mode_enabled', 'updated_at')
