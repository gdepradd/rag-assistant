import os
import io
import uuid
import base64
import time
import requests # Tambahkan impor requests
import fitz
from pypdf import PdfReader
from fastapi import FastAPI, Request, Response
import telebot
from telebot import apihelper
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==================== FIX SSL & NETWORK CLOUD ====================
apihelper.SESSION_TIME_TO_LIVE = 0
apihelper.CONNECT_TIMEOUT = 90
apihelper.READ_TIMEOUT = 90

# Inisialisasi FastAPI
app = FastAPI()

# Kredensial
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID")) if os.getenv("ADMIN_ID") else 0
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN") # Wajib ditambahkan di Railway

COLLECTION_NAME = "asisten_rag_konteks"
THRESHOLD = 0.60

# Instance
bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# ==================== HELPER: HUGGING FACE EMBEDDING API ====================

def get_embedding_from_hf(text: str) -> list:
    api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

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

# ==================== HELPER: GROQ VISION OCR ====================
def extract_text_with_vision_llm(image_bytes: bytes) -> str:
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    prompt = "Ekstrak seluruh teks dari gambar ini secara presisi. Jangan tambahkan komentar, pembuka, atau penjelasan. Kembalikan murni hanya teks."
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
            ],
            temperature=0.0,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception:
        return ""

# ==================== HANDLER TELEGRAM BOT ====================

@bot.message_handler(commands=['add'])
def handle_add_context(message):
    if message.from_user.id != ADMIN_ID: return
    try:
        args = message.text.split(' ', 2)
        if len(args) < 3: return
        source_tag, text_data = args[1], args[2]
        chunks = text_splitter.split_text(text_data)
        
        points = []
        for chunk in chunks:
            vector = get_embedding_from_hf(chunk) # Menggunakan API HF
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk, "source": source_tag}))
            time.sleep(0.5) # Jeda untuk menghindari Rate Limit HF API
            
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        bot.reply_to(message, f"✅ Berhasil menambahkan {len(chunks)} chunk.")
    except Exception as e:
        bot.reply_to(message, f"❌ Gagal: {e}")

@bot.message_handler(commands=['delete'])
def handle_delete_context(message):
    if message.from_user.id != ADMIN_ID: return
    try:
        source_tag = message.text.split(' ', 1)[1]
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_tag))]))
        bot.reply_to(message, f"🗑️ Data '{source_tag}' dihapus.")
    except Exception as e:
        bot.reply_to(message, f"❌ Gagal: {e}")

# Pastikan decorator menangkap tipe document dan photo
@bot.message_handler(content_types=['document', 'photo', 'text'])
def handle_admin_upload(message):
    if message.from_user.id != ADMIN_ID: 
        return

    # 1. Ekstrak teks pemicu (apakah dari teks biasa atau dari caption file)
    command_text = message.caption if message.content_type in ['document', 'photo'] else message.text
    
    # 2. Validasi apakah pesan mengandung perintah /add
    if not command_text or not command_text.startswith('/add'):
        return

    try:
        text_content, source_tag = "", ""
        
        # 3. Beri indikator bahwa bot menerima perintah dan sedang bekerja
        status_msg = bot.reply_to(message, "⏳ Mengunduh dan mengekstrak dokumen. Proses ini mungkin memakan waktu...")

        if message.content_type == 'document':
            file_name = message.document.file_name
            base_name, file_extension = os.path.splitext(file_name)
            source_tag = base_name.replace(" ", "_").lower()
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)

            if file_extension.lower() == '.txt':
                text_content = downloaded_file.decode('utf-8')
            elif file_extension.lower() == '.pdf':
                pdf_file = io.BytesIO(downloaded_file)
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted: text_content += extracted + "\n"
                
                # Jika PyPDF2 gagal/kosong, gunakan OCR Vision
                if not text_content.strip():
                    bot.edit_message_text("🔄 Mode OCR aktif. Menggunakan Groq Vision untuk membaca PDF...", chat_id=message.chat.id, message_id=status_msg.message_id)
                    pdf_document = fitz.open(stream=downloaded_file, filetype="pdf")
                    for page_num in range(len(pdf_document)):
                        pix = pdf_document.load_page(page_num).get_pixmap(dpi=150)
                        text_content += extract_text_with_vision_llm(pix.tobytes("jpeg")) + "\n"
                        
            elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
                text_content = extract_text_with_vision_llm(downloaded_file)
                
        elif message.content_type == 'photo':
            source_tag = f"photo_{message.message_id}"
            file_info = bot.get_file(message.photo[-1].file_id)
            text_content = extract_text_with_vision_llm(bot.download_file(file_info.file_path))

        # 4. Tangani silent failure jika teks tetap kosong
        if not text_content.strip(): 
            bot.edit_message_text("❌ Gagal mengekstrak teks. Dokumen kosong atau format tidak didukung.", chat_id=message.chat.id, message_id=status_msg.message_id)
            return
        
        bot.edit_message_text(f"✅ Teks berhasil diekstrak. Memulai proses embedding dan vectorisasi ke Qdrant...", chat_id=message.chat.id, message_id=status_msg.message_id)
        
        chunks = text_splitter.split_text(text_content)
        points = []
        for chunk in chunks:
            vector = get_embedding_from_hf(chunk) 
            points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk, "source": source_tag}))
            time.sleep(0.5) 

        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        bot.reply_to(message, f"✅ Selesai. File berhasil masuk database dengan Tag: `/delete {source_tag}`")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Terjadi kesalahan fatal: {e}")

@bot.message_handler(func=lambda message: True)
def handle_rag_query(message):
    user_question = message.text
    if user_question.startswith('/'): return

    try:
        query_vector = get_embedding_from_hf(user_question) # Menggunakan API HF
        search_result = qdrant_client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=3).points

        if not search_result or search_result[0].score < THRESHOLD:
            pesan =f"Informasi tidak diketahui atau Sistem tidak dirancang untuk menjawab pertanyaan tersebut\n {search_result[0].score}"
            bot.reply_to(message, pesan)
            return

        context_chunks = [hit.payload['text'] for hit in search_result if hit.score >= THRESHOLD]
        context_text = "\n---\n".join(context_chunks)

        system_prompt = f"""Anda adalah asisten AI yang bertugas menjawab pertanyaan berdasarkan konteks yang diberikan secara objektif dan to the point(diperbolehkan menambahkan sedikit kata pengantar dan penutup, selama tidak menyalahi informasi).
        Aturan ketat:
        1. Jawab pertanyaan HANYA menggunakan informasi dari teks Konteks di bawah ini.
        2. Jangan gunakan pengetahuan luar Anda yang tidak tertulis di dalam konteks.
        KONTEKS:\n{context_text}"""
        
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
            temperature=0.1,
            max_tokens=500
        )
        bot.reply_to(message, completion.choices[0].message.content)
    except Exception as e:
        bot.reply_to(message, f"Terjadi kesalahan: {e}")

# ==================== ROUTING FASTAPI (WEBHOOK) ====================

@app.post(f"/webhook/{TELEGRAM_TOKEN}")
async def telegram_webhook(request: Request):
    json_string = await request.json()
    update = telebot.types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return Response(status_code=200)

@app.on_event("startup")
def setup_webhook():
    railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
    if railway_domain:
        webhook_url = f"https://{railway_domain}/webhook/{TELEGRAM_TOKEN}"
        for attempt in range(3):
            try:
                bot.remove_webhook()
                time.sleep(1) 
                bot.set_webhook(url=webhook_url)
                print(f"✅ Webhook terpasang: {webhook_url}")
                break
            except Exception as e:
                time.sleep(5)