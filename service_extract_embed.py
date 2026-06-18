import os
import base64
import requests
import time
from config import groq_client, HF_API_TOKEN

# Embedding's model, api from HF use bge-m3
def get_embedding_from_hf(text: str) -> list:
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    max_retries = 3
    for attempt in range(max_retries):
        response = requests.post(api_url, headers=headers, json={"inputs": text})

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and isinstance(result[0], list):
                return result[0]
            return result
        elif "is currently loading" in response.text or response.status_code == 503:
            time.sleep(5)
            continue
        else:
            raise Exception(f"Error HF API: {response.text}")

    raise Exception("HF API gagal merespons setelah percobaan maksimal.")

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