import os
from dotenv import load_dotenv
import telebot
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Konfigurasi Global
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
COLLECTION_NAME = "asisten_rag_konteks"

# Validasi Kredensial Esensial
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN belum dikonfigurasi di environment!")
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY belum dikonfigurasi di environment!")

# Inisialisasi Client secara Global
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Pastikan Collection Qdrant ada dengan dimensi 768 (Gemini)
try:
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"[+] Membuat collection '{COLLECTION_NAME}' dengan dimensi 768...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
except Exception as e:
    print(f"[!] Gagal mengecek/membuat collection Qdrant: {e}")