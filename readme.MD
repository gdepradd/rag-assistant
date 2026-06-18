# Telegram RAG Assistant Bot

Bot Telegram berbasis Retrieval-Augmented Generation (RAG) yang dirancang untuk menjawab pertanyaan berdasarkan konteks dokumen yang diberikan oleh pengguna (Teks, PDF, atau Gambar). Bot ini dideploy menggunakan arsitektur Webhook di Railway.

---

## Tech Stack

* **Bahasa Pemrograman:** Python 3.10
* **Bot Framework:** pyTelegramBotAPI (Telebot)
* **Web Server (Webhook):** Flask
* **LLM (Inference & Vision):** Groq API

  * `meta-llama/llama-4-scout-17b-16e-instruct` (vision)
  * `llama-3.1-8b-instant` (generative text)
* **Embedding Model:** Google Gemini API (`gemini-embedding-2` via google-genai SDK)
* **Vector Database:** Qdrant Cloud
* **Text Processing:**

  * PyMuPDF (ekstraksi PDF)
  * langchain-text-splitters (chunking data)
* **Deployment:** Railway (via Dockerfile)

---

## Catatan Historis Infrastruktur (Embedding)

Pada pengembangan awal, proyek ini menggunakan Hugging Face Inference API (`all-MiniLM-L6-v2`) untuk embedding. Namun, pendekatan tersebut ditinggalkan karena kendala stabilitas (sering terjadi error HTTP 500/503) dan rate limit yang ketat (HTTP 429) pada tier gratis.

Penggunaan Ollama lokal juga dievaluasi, namun dibatalkan karena keterbatasan memori (RAM/VRAM) pada lingkungan deployment PaaS (Railway).

Saat ini, sistem menggunakan Google Gemini API yang menawarkan stabilitas uptime lebih baik untuk production dengan dimensi vektor 768.

---

## Alur Kerja Sistem (Workflow)

Sistem beroperasi dalam dua alur utama:

1. **Penyimpanan Pengetahuan (Ingestion)**
2. **Pengambilan Jawaban (Retrieval)**

---
## Diagram Alur Kerja (Flowchart)

```mermaid
graph TD
    classDef startEnd fill:#333,stroke:#fff,stroke-width:2px,color:#fff;
    classDef process fill:#1f77b4,stroke:#fff,stroke-width:1px,color:#fff;
    classDef decision fill:#ff7f0e,stroke:#fff,stroke-width:1px,color:#fff;
    classDef io fill:#2ca02c,stroke:#fff,stroke-width:1px,color:#fff;
    classDef db fill:#9467bd,stroke:#fff,stroke-width:1px,color:#fff;

    Start([START]):::startEnd --> Input[/INPUT <br> (Pesan Telegram)/]:::io
    Input --> isCommand{Diawali /add <br> ATAU /del?}:::decision

    isCommand -- "TIDAK" --> EmbedQuery[EMBEDDING QUERY <br> (Gemini API)]:::process
    EmbedQuery --> SearchQuery[PENCARIAN KEMIRIPAN]:::process
    SearchQuery --> dbQdrant1[(QDRANT DB)]:::db
    
    dbQdrant1 -.-> CheckThreshold{Skor Kemiripan <br> > Threshold?}:::decision

    CheckThreshold -- "TIDAK" --> ReturnUnknown[RETURN <br> 'Tidak Diketahui']:::process
    ReturnUnknown --> OutputRAGFail[/OUTPUT/]:::io
    OutputRAGFail --> End([END]):::startEnd

    CheckThreshold -- "YA" --> GenText[GENERATE TEXT <br> (Groq Llama 3)]:::process
    GenText --> OutputRAGSuccess[/OUTPUT/]:::io
    OutputRAGSuccess --> End

    isCommand -- "YA" --> isAdmin{User == <br> ADMIN_ID?}:::decision

    isAdmin -- "TIDAK" --> DenyCmd[PERINTAH DITOLAK]:::process
    DenyCmd --> OutputDeny[/OUTPUT/]:::io
    OutputDeny --> End

    isAdmin -- "YA" --> isDelete{Perintah <br> == /del?}:::decision

    isDelete -- "YA" --> DeleteCtx[DELETE CONTEXT]:::process
    DeleteCtx --> dbQdrant2[(QDRANT DB)]:::db
    dbQdrant2 -.-> OutputDel[/OUTPUT <br> ('Berhasil dihapus')/]:::io
    OutputDel --> End

    isDelete -- "TIDAK" --> isDoc{Ada Lampiran <br> GAMBAR/DOC?}:::decision

    isDoc -- "YA" --> Extract[EXTRACT WITH <br> Llama Vision / PyMuPDF]:::process
    Extract --> EmbedAdd[EMBEDDING <br> (Gemini API)]:::process
    isDoc -- "TIDAK" --> EmbedAdd

    EmbedAdd --> UploadDB[UPLOAD VECTOR DB]:::process
    UploadDB --> dbQdrant3[(QDRANT DB)]:::db
    dbQdrant3 -.-> OutputAdd[/OUTPUT <br> ('Berhasil ditambah')/]:::io
    OutputAdd --> End
```

## 1. Alur Penyimpanan Data (`/add`)

### Input Pengguna

Admin mengirimkan data berupa:

* Teks langsung
* File PDF
* Gambar

### Ekstraksi Teks

* **PDF** diproses menggunakan PyMuPDF
* **Gambar** dianalisis menggunakan vision model via Groq API untuk mengekstrak informasi menjadi format JSON terstruktur

### Chunking

Teks panjang dipecah menjadi potongan kecil (**chunks**) menggunakan `RecursiveCharacterTextSplitter`.

Konfigurasi:

* Chunk size: **500**
* Overlap: **50**

Tujuan:

* Menjaga konteks tetap relevan
* Mempermudah retrieval semantik

### Vectorization

Setiap chunk diubah menjadi dense vector berdimensi **768** menggunakan Google Gemini API.

### Penyimpanan

Vektor dan metadata (teks asli) disimpan ke **Qdrant Cloud**.

---

## 2. Alur Tanya Jawab (Chatbot)

### Input Pengguna

Pengguna mengirimkan pertanyaan ke bot Telegram.

### Vectorization Pertanyaan

Pertanyaan diubah menjadi vektor menggunakan Gemini API.

### Pencarian Konteks (Similarity Search)

Sistem mencari chunk di Qdrant yang memiliki kedekatan jarak (**Cosine Similarity**) paling tinggi dengan vektor pertanyaan.

### Generasi Jawaban

Konteks yang ditemukan digabungkan dengan pertanyaan asli pengguna dan dikirim ke Groq API (**Llama 3**) sebagai prompt sistem.

### Output

Bot mengirimkan jawaban akhir yang relevan dan berbasis konteks ke pengguna.

---

## Cara Menjalankan (Lokal)

### 1. Clone Repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Buat Virtual Environment dan Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Buat File `.env`

Isi file `.env` di folder utama dengan kredensial berikut:

```env
ENVIRONMENT=local
TELEGRAM_BOT_TOKEN=your_telegram_token
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
ADMIN_ID=your_telegram_user_id
```

### 4. Jalankan Bot

```bash
python app.py
```
