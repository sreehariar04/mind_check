from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from .models import JournalEntry, UserPreference


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(required=True)
    last_name = forms.CharField(required=True)
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email', 'password1', 'password2')

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError('This email is already registered.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        email = self.cleaned_data.get('email', '').strip().lower()
        user.username = email
        user.email = email
        if commit:
            user.save()
        return user


class JournalEntryForm(forms.ModelForm):
    class Meta:
        model = JournalEntry
        fields = ('content',)
        widgets = {
            'content': forms.Textarea(
                attrs={
                    'rows': 8,
                    'placeholder': 'Write your thoughts for today...'
                }
            )
        }


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ('first_name', 'last_name', 'email')

    def clean_email(self):
        email = self.cleaned_data.get('email', '').strip().lower()
        if not email:
            raise forms.ValidationError('Email is required.')

        queryset = User.objects.filter(username=email)
        if self.instance.pk:
            queryset = queryset.exclude(pk=self.instance.pk)
        if queryset.exists():
            raise forms.ValidationError('This email is already registered.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        normalized_email = self.cleaned_data['email']
        user.email = normalized_email
        user.username = normalized_email
        if commit:
            user.save()
        return user


class UserPreferenceForm(forms.ModelForm):
    class Meta:
        model = UserPreference
        fields = ('notifications_enabled', 'dark_mode_enabled')


class EmailAuthenticationForm(AuthenticationForm):
    username = forms.CharField(
        label='Email',
        widget=forms.EmailInput(attrs={'autofocus': True})
    )