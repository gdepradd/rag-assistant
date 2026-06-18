# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set environment variables agar log lebih bersih
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set direktori kerja
WORKDIR /app

# Install dependency sistem untuk build library tertentu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua source code
COPY . .

# Eksekusi app.py langsung
CMD ["python", "app.py"]