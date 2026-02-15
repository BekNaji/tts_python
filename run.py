import os
import sys
import torch
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify
import time
import re
from num2words import num2words

# Flutter stdout'dan real vaqtda o'qishi uchun flushni majburiy qilamiz
def log(message):
    print(f"--- LOG: {message} ---", flush=True)

app = Flask(__name__)

# --- KONFIGURATSIYA ---
torch.set_num_threads(os.cpu_count())
torch.set_grad_enabled(False)
device = torch.device('cpu')
SAMPLE_RATE = 48000 

LANG_CONFIG = {
    'uz': {'file': 'silero_uz_model.pt', 'speaker': 'dilnavoz'},
    'ru': {'file': 'silero_ru_model.pt', 'speaker': 'aidar'},
    'en': {'file': 'silero_en_model.pt', 'speaker': 'en_0'},
}

MODELS = {}

# --- MATNNI BO'LAKLARGA BO'LISH (CHUNKING) ---
def split_text(text, max_chars=700):
    """Matnni mantiqiy gaplarga bo'ladi, model qiynalmasligi uchun."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- MATNNI NORMALLASHTIRISH ---
def uz_num2words(text):
    units = ["", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
    tens = ["", "o'n", "yigirma", "o'ttiz", "qirq", "ellik", "oltmish", "yetmish", "sakson", "to'qson"]
    
    def convert_chunk(n):
        res = ""
        if n >= 100: res += units[n // 100] + " yuz "; n %= 100
        if n >= 10: res += tens[n // 10] + " "; n %= 10
        if n > 0: res += units[n] + " "
        return res.strip()

    def replace_num(match):
        num = int(match.group())
        if num == 0: return "nol"
        parts = []
        if num >= 1000000000: 
            parts.append(convert_chunk(num // 1000000000) + " milliard")
            num %= 1000000000
        if num >= 1000000: 
            parts.append(convert_chunk(num // 1000000) + " million")
            num %= 1000000
        if num >= 1000: 
            parts.append(convert_chunk(num // 1000) + " ming")
            num %= 1000
        if num > 0: 
            parts.append(convert_chunk(num))
        return " ".join(parts).strip()

    return re.sub(r'\d+', replace_num, text)

def normalize_text(text, lang):
    if not text: return ""
    log(f"Matn normallashtirilmoqda: {lang}")
    try:
        if lang == 'uz':
            return uz_num2words(text)
        else:
            def replace_num(match):
                return num2words(int(match.group()), lang=lang)
            return re.sub(r'\d+', replace_num, text)
    except Exception as e:
        log(f"Normalization warning: {e}")
        return text

# --- MODELLARNI YUKLASH ---
def get_full_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.getcwd(), filename)

log("Modellar yuklanmoqda...")
for lang, cfg in LANG_CONFIG.items():
    path = get_full_path(cfg['file'])
    if os.path.exists(path):
        try:
            log(f"Loading {lang.upper()} model...")
            importer = torch.package.PackageImporter(path)
            model = importer.load_pickle("tts_models", "model")
            model.to(device)
            MODELS[lang] = model
        except Exception as e:
            log(f"XATO: {lang} yuklanmadi: {e}")
    else:
        log(f"WARNING: {path} topilmadi.")

log("Model tayyor bo'ldi")

# --- API ENDPOINT ---
@app.route('/tts', methods=['POST'])
def tts_engine():
    try:
        data = request.json
        raw_text = data.get('text', '')
        lang = data.get('lang', 'uz')
        speaker = data.get('speaker', LANG_CONFIG.get(lang, {}).get('speaker', 'dilnavoz'))
        output_path = data.get('file', 'output.wav')

        if lang not in MODELS:
            return jsonify({"status": "error", "message": f"{lang} modeli yuklanmagan"}), 400

        # 1. Tayyorgarlik
        clean_text = normalize_text(raw_text, lang)
        text_chunks = split_text(clean_text)
        
        log(f"Jarayon boshlandi: {len(text_chunks)} ta bo'lak.")
        
        all_audio = []
        start_time = time.time()

        # 2. Ketma-ket generatsiya
        for idx, chunk in enumerate(text_chunks):
            if not chunk.strip(): continue
            log(f"Generatsiya {idx+1}/{len(text_chunks)}...")
            with torch.inference_mode():
                audio = MODELS[lang].apply_tts(
                    text=chunk,
                    speaker=speaker,
                    sample_rate=SAMPLE_RATE
                )
                all_audio.append(audio.numpy())

        # 3. Faylga yozish
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            sf.write(output_path, combined_audio, SAMPLE_RATE)
            
            duration = len(combined_audio) / SAMPLE_RATE
            gen_time = time.time() - start_time
            
            log(f"Tayyor! Audio davomiyligi: {duration:.2f}s, Generatsiya vaqti: {gen_time:.2f}s")
            
            return jsonify({
                "status": "success",
                "file": output_path,
                "duration": f"{duration:.2f}",
                "gen_time": f"{gen_time:.2f}"
            })
        else:
            return jsonify({"status": "error", "message": "Matn bo'sh"}), 400

    except Exception as e:
        log(f"API ERROR: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
    log(f"Server 127.0.0.1:{port} portida ishga tushdi")
    app.run(port=port, host='127.0.0.1', debug=False)