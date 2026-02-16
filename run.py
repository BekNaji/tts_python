import os
import sys
import torch
import soundfile as sf
import numpy as np
from flask import Flask, request, jsonify
import time
import re
import requests
from num2words import num2words

def log(message):
    print(f"--- LOG: {message} ---", flush=True)

app = Flask(__name__)

# --- KONFIGURATSIYA ---
torch.set_num_threads(4)
torch.set_grad_enabled(False)
device = torch.device('cpu')
SAMPLE_RATE = 48000 

LANG_CONFIG = {
    'uz': {
        'file': 'silero_uz_model.pt', 
        'url': 'https://models.silero.ai/models/tts/uz/v4_uz.pt', 
        'speaker': 'dilnavoz'
    },
    'ru': {
        'file': 'v5_cis_base_nostress.pt', 
        'url': 'https://models.silero.ai/models/tts/ru/v5_cis_base.pt', 
        'speaker': 'aidar'
    },
    'en': {
        'file': 'silero_en_model.pt', 
        'url': 'https://models.silero.ai/models/tts/en/v3_en.pt', 
        'speaker': 'en_0'
    },
}

MODELS = {}

def get_full_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.getcwd(), filename)

# --- MODELNI YUKLAB OLISH (XAVFSIZ USUL) ---
def download_model(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000000: # 1MB dan katta bo'lsa bor deb hisoblaymiz
        return True
    
    log(f"Model yuklab olinmoqda: {url}")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        log(f"Yuklab olindi: {save_path}")
        return True
    except Exception as e:
        log(f"Yuklashda xato: {url} -> {e}")
        if os.path.exists(save_path): os.remove(save_path)
        return False

# --- MODELLARNI YUKLASH ---
log("Modellarni yuklash boshlandi...")
for lang, cfg in LANG_CONFIG.items():
    path = get_full_path(cfg['file'])
    if download_model(cfg['url'], path):
        try:
            log(f"{lang.upper()} model xotiraga yuklanmoqda...")
            importer = torch.package.PackageImporter(path)
            model = importer.load_pickle("tts_models", "model")
            model.to(device)
            MODELS[lang] = model
            log(f"MUVAFFAQIYAT: {lang.upper()} modeli tayyor!")
        except Exception as e:
            log(f"XATO: {lang} modelini o'qib bo'lmadi: {e}")
    else:
        log(f"OGOHLANTIRISH: {lang} modeli o'tkazib yuborildi.")

# --- MATN FUNKSIYALARI (O'ZGARISHSIZ) ---
def split_text(text, max_chars=700):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    curr = ""
    for s in sentences:
        if len(curr) + len(s) < max_chars: curr += s + " "
        else:
            chunks.append(curr.strip())
            curr = s + " "
    if curr: chunks.append(curr.strip())
    return chunks

def uz_num2words(text):
    # Sizning uz_num2words funksiyangiz shu yerda bo'lsin
    return text 

def normalize_text(text, lang):
    if not text: return ""
    try:
        if lang == 'uz': return uz_num2words(text)
        else: return re.sub(r'\d+', lambda m: num2words(int(m.group()), lang=lang), text)
    except: return text

@app.route('/tts', methods=['POST'])
def tts_engine():
    data = request.json
    lang = data.get('lang', 'uz')
    log(f"Yangi so'rov: lang={lang}, text_length={len(data.get('text', ''))}")
    if lang not in MODELS:
        return jsonify({"status": "error", "message": f"{lang} modeli yuklanmagan. Jami: {list(MODELS.keys())}"}), 400
    
    try:
        raw_text = data.get('text', '')
        speaker = data.get('speaker', LANG_CONFIG[lang]['speaker'])
        output_path = data.get('file', 'output.wav')
        
        clean_text = normalize_text(raw_text, lang)
        chunks = split_text(clean_text)
        
        all_audio = []
        for chunk in chunks:
            if not chunk.strip(): continue
            
            # Nutqni sekinlashtirish (standart tezlikdan 75% yoki "slow" - sekin)
            # Masalan: 1.0 - normal, 0.7 - sekinroq, 1.2 - tezroq
            # SSML tegi orqali tezlikni boshqaramiz
            ssml_text = f'<prosody rate="0.8">{chunk}</prosody>'
            
            log(f"Generatsiya qilinmoqda (tezlik: 0.8)...")
            audio = MODELS[lang].apply_tts(
                text=ssml_text, 
                speaker=speaker, 
                sample_rate=SAMPLE_RATE
            )
            all_audio.append(audio.numpy())
            
        if all_audio:
            sf.write(output_path, np.concatenate(all_audio), SAMPLE_RATE)
            return jsonify({"status": "success", "file": output_path})
        return jsonify({"status": "error", "message": "Bo'sh matn"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5005
    app.run(port=port, host='127.0.0.1')