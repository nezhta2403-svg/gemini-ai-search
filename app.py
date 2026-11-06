import os
import time
from collections import defaultdict, deque

from google import genai
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL = "gemini-3.5-flash-lite"
MAX_QUERY_LEN = 500
RATE_LIMIT = 10
RATE_WINDOW = 60

SYSTEM_PROMPT = """
Ты — умный ИИ-ассистент.
Отвечай на языке пользователя.
Давай точные, понятные и полезные ответы.
Не выдумывай факты.
Если не знаешь ответа, честно скажи об этом.
Используй структурированные ответы.
"""

app = Flask(__name__)

hits = defaultdict(deque)


def too_many_requests(ip):
    now = time.time()
    q = hits[ip]

    while q and now - q[0] > RATE_WINDOW:
        q.popleft()

    if len(q) >= RATE_LIMIT:
        return True

    q.append(now)
    return False


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.post("/api/search")
def search():
    ip = request.remote_addr or "?"

    if too_many_requests(ip):
        return jsonify(error="Слишком много запросов. Подождите минуту."), 429

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify(error="Введите вопрос."), 400

    if len(query) > MAX_QUERY_LEN:
        return jsonify(error="Слишком длинный вопрос."), 400

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify(error="Не задан API-ключ Gemini."), 500

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL,
            contents=query,
            config={
                "system_instruction": SYSTEM_PROMPT
            }
        )

        return jsonify(
            answer=response.text or "Не удалось получить ответ.",
            sources=[]
        )

    except Exception:
        app.logger.exception("Gemini API error")
        return jsonify(
            error="Ошибка при обращении к Gemini API."
        ), 502


if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Задайте GEMINI_API_KEY.")

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )
