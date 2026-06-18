Telegram RAG Assistant Bot

Bot Telegram berbasis Retrieval-Augmented Generation (RAG) yang dirancang untuk menjawab pertanyaan berdasarkan konteks dokumen yang diberikan oleh pengguna (Teks, PDF, atau Gambar). Bot ini dideploy menggunakan arsitektur Webhook di Railway.

Tech Stack

Bahasa Pemrograman: Python 3.10

Bot Framework: pyTelegramBotAPI (Telebot)

Web Server (Webhook): Flask

LLM (Inference & Vision): Groq API (meta-llama/llama-4-scout-17b-16e-instruct(vision) & llama-3.1-8b-instant(generative text) )

Embedding Model: Google Gemini API (gemini-embedding-2 via google-genai SDK)

Vector Database: Qdrant Cloud

Text Processing: PyMuPDF (ekstraksi PDF) & langchain-text-splitters (chunking data)

Deployment: Railway (via Dockerfile)

Catatan Historis Infrastruktur (Embedding)

Pada pengembangan awal, proyek ini menggunakan Hugging Face Inference API (all-MiniLM-L6-v2) untuk embedding. Namun, pendekatan tersebut ditinggalkan karena kendala stabilitas (sering terjadi error HTTP 500/503) dan rate limit yang ketat (HTTP 429) pada tier gratis. Penggunaan Ollama lokal juga dievaluasi, namun dibatalkan karena keterbatasan memori (RAM/VRAM) pada lingkungan deployment PaaS (Railway). Saat ini, sistem menggunakan Google Gemini API yang menawarkan stabilitas uptime lebih baik untuk production dengan dimensi vektor 768.

Alur Kerja Sistem (Workflow)

Sistem beroperasi dalam dua alur utama: Penyimpanan Pengetahuan (Ingestion) dan Pengambilan Jawaban (Retrieval).

1. Alur Penyimpanan Data (/add)

Input Pengguna: Admin mengirimkan data berupa teks langsung, file PDF, atau Gambar.

Ekstraksi Teks: * PDF diproses menggunakan PyMuPDF.

Gambar dianalisis menggunakan vision model via Groq API untuk mengekstrak informasi menjadi format JSON terstruktur.

Chunking: Teks panjang dipecah menjadi potongan kecil (chunks) menggunakan RecursiveCharacterTextSplitter (ukuran 500, overlap 50) agar konteks tetap relevan.

Vectorization: Setiap chunk diubah menjadi angka (dense vector berdimensi 768) menggunakan Google Gemini API.

Penyimpanan: Vektor dan metadata (teks asli) disimpan ke Qdrant Cloud.

2. Alur Tanya Jawab (Chatbot)

Input Pengguna: Pengguna mengirimkan pertanyaan ke bot Telegram.

Vectorization Pertanyaan: Pertanyaan diubah menjadi vektor menggunakan Gemini API.

Pencarian Konteks (Similarity Search): Sistem mencari chunk di Qdrant yang memiliki kedekatan jarak (Cosine Similarity) paling tinggi dengan vektor pertanyaan.

Generasi Jawaban: Konteks yang ditemukan digabungkan dengan pertanyaan asli pengguna dan dikirim ke Groq API (Llama 3) sebagai prompt sistem.

Output: Bot mengirimkan jawaban akhir yang relevan dan berbasis konteks ke pengguna.

Cara Menjalankan (Lokal)

Clone repositori ini.

Buat virtual environment dan install dependensi:

pip install -r requirements.txt


Buat file .env di folder utama dan isi dengan kredensial berikut:

ENVIRONMENT=local
TELEGRAM_BOT_TOKEN=your_telegram_token
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
ADMIN_ID=your_telegram_user_id


Jalankan bot:

python app.py
