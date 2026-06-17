import os
from dotenv import load_dotenv
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Konfigurasi Kredensial (Ganti dengan data dari Tahap 1)
# Memuat variabel dari file .env
load_dotenv()

# Tarik konfigurasi langsung dari environment variable
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "asisten_rag_konteks"
# 2. Inisialisasi Qdrant Client & Embedding Model
print("Memuat model embedding BAAI/bge-m3...")
model = SentenceTransformer('BAAI/bge-m3') # Dimensi output: 1024
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Tambahkan import ini di bagian atas file ingest.py jika belum ada
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType

def init_collection():
    """Membuat koleksi dan indeks payload di Qdrant jika belum ada."""
    collections = client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    
    if not exists:
        print(f"Membuat koleksi baru: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        
        # --- TAMBAHKAN BLOK KODE INI ---
        print(f"Membuat payload index untuk kunci 'source'...")
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        # -------------------------------
    else:
        print(f"Koleksi '{COLLECTION_NAME}' sudah tersedia.")
        
        # Jaga-jaga jika koleksi sudah ada tapi indeks belum dibuat, 
        # kita paksa buat di sini agar tidak error saat delete.
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Jika indeks sudah ada, Qdrant akan melempar error ringan, kita abaikan saja
            pass
def add_context(text_data: str, source_tag: str):
    """Memproses teks, membuat embedding, dan upload ke Qdrant."""
    init_collection()
    
    # Text Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text_data)
    print(f"Teks dipecah menjadi {len(chunks)} chunks.")
    
    points = []
    for chunk in chunks:
        # Generate Embedding untuk tiap chunk
        vector = model.encode(chunk).tolist()
        point_id = str(uuid.uuid4())
        
        # Bungkus ke dalam struktur poin Qdrant
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "source": source_tag # Digunakan sebagai identifier saat menghapus
                }
            )
        )
        
    # Upsert ke Qdrant Cloud
    print(f"Mengunggah {len(points)} vektor ke Qdrant Cloud...")
    operation_info = client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    print("Ingestion berhasil!", operation_info)

if __name__ == "__main__":
    # Contoh Data Konteks Lama/Baru
    sample_context = (
        "Jadwal operasional layanan konsultasi IT adalah setiap hari Senin hingga Jumat "
        "mulai pukul 09:00 WIB sampai 17:00 WIB. Layanan libur pada hari Sabtu, Minggu, "
        "dan hari libur nasional. Untuk kontak darurat di luar jam kerja bisa hubungi admin@email.com."
    )
    # Sumber tag harus unik per dokumen/topik untuk memudahkan manajemen data
    tag = "jadwal_operasional" 
    
    add_context(sample_context, tag)