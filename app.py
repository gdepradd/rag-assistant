import os
import sys
from config import bot
from msg_telebot_handler import register_handlers

# Daftarkan semua handler pesan Telegram sebelum bot berjalan
register_handlers()

def run_production():
    """
    Menjalankan bot menggunakan Flask Webhook untuk lingkungan produksi (Railway).
    """
    from flask import Flask, request
    import telebot

    server = Flask(__name__)
    port = int(os.environ.get('PORT', 5000))
    
    # Memformat URL Webhook dari variabel lingkungan Railway
    raw_railway_url = os.environ.get('RAILWAY_STATIC_URL', '')
    if raw_railway_url and not raw_railway_url.startswith("http"):
        webhook_url = f"https://{raw_railway_url}"
    else:
        webhook_url = raw_railway_url

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not webhook_url or not bot_token:
        print("[!] ERROR: RAILWAY_STATIC_URL atau TELEGRAM_BOT_TOKEN tidak dikonfigurasi!")
        sys.exit(1)

    @server.route(f"/{bot_token}", methods=['POST'])
    def get_updates():
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return "!", 200

    @server.route("/")
    def check_webhook():
        bot.remove_webhook()
        bot.set_webhook(url=f"{webhook_url}/{bot_token}")
        return "RAG Bot Webhook Aktif", 200

    print(f"[!] Menjalankan Webhook pada port {port} dengan URL: {webhook_url}")
    server.run(host="0.0.0.0", port=port)

def run_local():
    """
    Menjalankan bot menggunakan Polling untuk pengujian lokal.
    """
    print("[!] Menjalankan bot dalam mode Lokal (Polling)...")
    bot.remove_webhook()
    bot.infinity_polling(timeout=10, long_polling_timeout=5)

if __name__ == "__main__":
    env_mode = os.getenv("ENVIRONMENT", "local")

    if env_mode == "production":
        run_production()
    else:
        run_local()