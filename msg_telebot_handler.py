import os
import io
import time
import uuid
import json
import fitz
from pypdf import PdfReader
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from config import bot, qdrant_client, groq_client, ADMIN_ID, COLLECTION_NAME,text_splitter
from service_extract_embed import extract_text_with_vision_llm, get_embedding_from_hf
THRESHOLD=0.50 # mengatur nilai treshold cos sim. 

def register_handlers():
    #Fitur add information ke database Qdrant, hanya bisa diakses oleh admin
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
    #fitur delete information dari database Qdrant, hanya bisa diakses oleh admin
    @bot.message_handler(commands=['delete'])
    def handle_delete_context(message):
        if message.from_user.id != ADMIN_ID: return
        try:
            source_tag = message.text.split(' ', 1)[1]
            qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source_tag))]))
            bot.reply_to(message, f"🗑️ Data '{source_tag}' dihapus.")
        except Exception as e:
            bot.reply_to(message, f"❌ Gagal: {e}")

    #fitur proses jika input add adalah dokumen atau foto.
    # Pastikan decorator menangkap tipe document dan photo
    @bot.message_handler(content_types=['document', 'photo'])
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
    #fitur RAQ untuk menjawab pertanyaan user berdasarkan konteks yang ada di database Qdrant
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

            system_prompt = f"""Anda adalah asisten AI yang bertugas menjawab pertanyaan berdasarkan konteks yang diberikan secara objektif dan to the point(diperbolehkan menambahkan sedikit kata pengantar dan penutup seperti ya atau tidak, selama tidak menyalahi informasi).
            Aturan ketat:
            1. Jawab pertanyaan HANYA menggunakan informasi dari teks Konteks di bawah ini.
            2. Jangan gunakan pengetahuan luar Anda yang tidak tertulis di dalam konteks.
            KONTEKS:\n{context_text}"""
            
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_question}],
                temperature=0.2,
                max_tokens=500
            )
            bot.reply_to(message, completion.choices[0].message.content)
        except Exception as e:
            bot.reply_to(message, f"Terjadi kesalahan: {e}")