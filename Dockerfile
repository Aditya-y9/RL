FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt pydantic openai fastapi uvicorn
COPY . .
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
