# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set environment variables agar Python tidak menulis file .pyc dan langsung mencetak log
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set folder kerja di dalam kontainer Docker
WORKDIR /app

# Install dependensi sistem yang dibutuhkan oleh PyMuPDF (fitz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements terlebih dahulu (untuk memanfaatkan Docker layer caching)
COPY requirements.txt .

# Install dependensi Python
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode program dari folder lokal ke folder /app di kontainer
# Ini akan membawa app.py, config.py, msg_telebot_handler.py, dan service_extract_embed.py
COPY . .

# Port yang akan digunakan oleh Flask Webhook (Railway akan otomatis menyediakan port ini via env)
EXPOSE 5000

# Jalankan file utama program
CMD ["python", "app.py"]