FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Avoid creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Run main.py (Cerebrium auto-triggers this)
CMD ["python", "main.py"]
