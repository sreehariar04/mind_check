from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from .forms import ProfileForm
from .models import EmotionResult, JournalEntry


class ProfileFormTests(TestCase):
	def test_profile_form_syncs_username_with_email(self):
		user = User.objects.create_user(
			username='old@example.com',
			email='old@example.com',
			password='StrongPass123!',
		)

		form = ProfileForm(
			data={
				'first_name': 'Test',
				'last_name': 'User',
				'email': 'new@example.com',
			},
			instance=user,
		)

		self.assertTrue(form.is_valid(), form.errors)
		updated_user = form.save()
		self.assertEqual(updated_user.email, 'new@example.com')
		self.assertEqual(updated_user.username, 'new@example.com')

	def test_profile_form_rejects_duplicate_email(self):
		User.objects.create_user(username='taken@example.com', email='taken@example.com', password='StrongPass123!')
		user = User.objects.create_user(username='other@example.com', email='other@example.com', password='StrongPass123!')

		form = ProfileForm(
			data={
				'first_name': 'Other',
				'last_name': 'User',
				'email': 'taken@example.com',
			},
			instance=user,
		)

		self.assertFalse(form.is_valid())
		self.assertIn('email', form.errors)


class HistoryPageTests(TestCase):
	def setUp(self):
		self.user = User.objects.create_user(
			username='tester@example.com',
			email='tester@example.com',
			password='StrongPass123!',
		)

	def test_delete_preserves_all_active_filters(self):
		entry = JournalEntry.objects.create(user=self.user, content='Feeling very happy today')
		EmotionResult.objects.create(entry=entry, emotion=EmotionResult.EMOTION_HAPPINESS, confidence=0.9)

		today = timezone.localdate().isoformat()
		self.client.force_login(self.user)

		response = self.client.post(
			reverse('history'),
			data={'entry_id': entry.id},
			QUERY_STRING=f'q=happy&emotion={EmotionResult.EMOTION_HAPPINESS}&date={today}',
		)

		self.assertEqual(response.status_code, 302)
		self.assertEqual(
			response.url,
			f"{reverse('history')}?q=happy&emotion={EmotionResult.EMOTION_HAPPINESS}&date={today}",
		)
