function getCookie(name) {
    const cookieValue = document.cookie
        .split('; ')
        .find((row) => row.startsWith(name + '='));
    return cookieValue ? decodeURIComponent(cookieValue.split('=')[1]) : null;
}

const form = document.getElementById('predict-form');
const textArea = document.getElementById('predict-text');
const loadingEl = document.getElementById('predict-loading');
const resultCard = document.getElementById('predict-result');
const resultEmotion = document.getElementById('result-emotion');
const resultConfidence = document.getElementById('result-confidence');
const resultMessage = document.getElementById('result-message');

if (textArea) {
    const autoGrow = () => {
        textArea.style.height = 'auto';
        textArea.style.height = `${textArea.scrollHeight}px`;
    };
    textArea.addEventListener('input', autoGrow);
    autoGrow();
}

if (form) {
    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const url = form.dataset.url;
        const content = textArea.value.trim();
        if (!content) {
            return;
        }

        loadingEl.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const payload = new FormData();
        payload.append('content', content);

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCookie('csrftoken') || '' },
            body: payload
        });

        loadingEl.classList.add('hidden');

        if (!response.ok) {
            resultEmotion.textContent = 'Unable to analyze right now.';
            resultConfidence.textContent = '';
            resultMessage.textContent = 'Please try again in a moment.';
            resultCard.classList.remove('hidden');
            return;
        }

        const data = await response.json();
        resultEmotion.textContent = data.emotion;
        resultConfidence.textContent = `Confidence: ${data.confidence}%`;
        resultMessage.textContent = data.message;
        resultCard.classList.remove('hidden');
    });
}
