# AI-Based Mental Health Monitoring System (Django)

This is a beginner-friendly, presentation-ready web application for tracking journal-based emotional wellness.

## Tech Stack

- Django (backend + templates)
- HTML, CSS, JavaScript
- SQLite (default Django DB)
- Chart.js (insights chart)

## Project Structure

```text
mind_check/
├─ manage.py
├─ db.sqlite3 (created after migrate)
├─ README.md
├─ mind_check_project/
│  ├─ settings.py
│  ├─ urls.py
│  └─ ...
├─ monitor/
│  ├─ admin.py
│  ├─ forms.py
│  ├─ ml_model.py
│  ├─ models.py
│  ├─ urls.py
│  ├─ views.py
│  └─ migrations/
├─ templates/
│  ├─ base.html
│  ├─ registration/login.html
│  └─ monitor/
│     ├─ landing.html
│     ├─ register.html
│     ├─ dashboard.html
│     ├─ journal.html
│     ├─ insights.html
│     ├─ history.html
│     └─ profile.html
└─ static/
   └─ css/style.css
```

## Setup Commands

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install django
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open: `http://127.0.0.1:8000/`

## Implemented Features

- User registration and login/logout
- Journal submission page
- Emotion prediction flow via `monitor/ml_model.py`
- Dashboard with latest emotion and recent entries
- Insights page with emotion counts, weekly summary, and 14-day trend chart
- History page with all previous entries and emotion tags
- Profile page with editable basic user details

## ML Integration (Later)

Update only `monitor/ml_model.py`:

1. Load your trained model (`.pkl`, Hugging Face model, etc.)
2. Replace placeholder logic in `predict_emotion(text)`
3. Return exactly:
   - emotion label (`Happiness`, `Sadness`, `Fear`, `Anger`, `Anxiety`, `Neutral`)
   - confidence score (`0.0` to `1.0`)

Everything else in the app will continue working without major changes.

## Deployment (Simple Path)

For academic deployment, easiest options:

1. **Render** (quick and free tier friendly)
2. **PythonAnywhere** (easy Django hosting)

Basic production checklist:

- Set `DEBUG = False`
- Set `ALLOWED_HOSTS`
- Keep `SECRET_KEY` in environment variable
- Run migrations on server
- Configure static files (`collectstatic`) if needed by host
