import os
import base64
import time
from config import groq_client
from google import genai
from google.genai import types

# Konfigurasi Gemini dengan library baru
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di environment variables.")

# Inisialisasi client Gemini yang baru
gemini_client = genai.Client(api_key=gemini_api_key)

def get_embedding_from_hf(text: str) -> list:
    """
    Menghasilkan dense vector embedding menggunakan Google GenAI SDK terbaru.
    """
    clean_text = text.replace("\n", " ").strip()
    
    for attempt in range(3):
        try:
            # Format pemanggilan untuk library google-genai versi terbaru
            response = gemini_client.models.embed_content(
                model='gemini-embedding-2',
                config=types.EmbedContentConfig(output_dimensionality=768),
                contents=clean_text,
            )
            return response.embeddings[0].values
        except Exception as e:
            if attempt == 2:
                raise Exception(f"Gagal generate embedding via Gemini API: {str(e)}")
            time.sleep(1.0)
#function for extract doc and image, use groq API platform, model : meta-llama/llama-4-scout-17b-16e-instruct
def extract_text_with_vision_llm(image_bytes: bytes) -> str:
   base64_image = base64.b64encode(image_bytes).decode('utf-8')
   prompt = """
    Ekstrak seluruh informasi, jadwal, dan tabel dari gambar ini.
    Kamu WAJIB mengembalikan data murni HANYA dalam format JSON Array of Objects.
    Gunakan key yang relevan (contoh: "kegiatan", "tanggal", "keterangan").
    Jika bukan berupa tabel, jadikan satu object JSON dengan key "informasi".
    DILARANG KERAS memberikan teks pembuka, penutup, atau penjelasan di luar kode JSON.
    
    Contoh Output yang Diharapkan:
    [
      {"kegiatan": "Wisuda 173", "tanggal": "14 Agustus 2024"},
      {"kegiatan": "Masa Pembayaran UKT", "tanggal": "2-10 Januari 2025"}
    ]
    atau contoh lain 
    [
      {"Nama Dosen": "Alexander", "NIP": "0012321"},
      {"Nama Dosen": "Hirohito", "NIP": "2203123"}
    ]
    """
   try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            temperature=0.0,
            max_tokens=4096 
        )
        
        raw_response = completion.choices[0].message.content
        # Menghapus blok markdown bawaan LLM dengan aman
        clean_json = raw_response.replace('```json', '').replace('```', '').strip()
        return clean_json
        
   except Exception as e:
        return f"[ERROR_VISION: {str(e)}]"