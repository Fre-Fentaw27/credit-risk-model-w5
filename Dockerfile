FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for MLflow and sklearn
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy MLflow database file
COPY mlflow.db .

COPY . .

# Ensure the MLflow tracking URI is set
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]