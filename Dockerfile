# Base image
FROM python:3.13-slim

# System deps for xgboost/scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . .

# Ensure artifacts dir exists
RUN mkdir -p artifacts

EXPOSE 80

CMD ["python", "app.py"]

