import os
from dotenv import load_dotenv
import telebot
from groq import Groq
from qdrant_client import QdrantClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# Konfigurasi Global
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
COLLECTION_NAME = "asisten_rag_konteks"
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# Validasi Kredensial Esensial
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise ValueError("TELEGRAM_BOT_TOKEN belum dikonfigurasi di environment!")

# Inisialisasi Client secara Global
bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
