from django.urls import path

from . import views


urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('register/', views.register_page, name='register'),
    path('dashboard/', views.dashboard_page, name='dashboard'),
    path('journal/', views.journal_page, name='journal'),
    path('predict/', views.predict_emotion_api, name='predict_emotion'),
    path('results/<int:entry_id>/', views.emotional_results_page, name='emotional_results'),
    path('insights/', views.insights_page, name='insights'),
    path('history/', views.history_page, name='history'),
    path('history/<str:selected_date>/', views.history_by_date, name='history_by_date'),
    path('settings/', views.settings_page, name='settings'),
    path('profile/', views.profile_page, name='profile'),
]