import torch
import os

# Kerakli parametrlar
language = 'ru'  # V5 CIS modellari ko'pincha 'ru' yoki 'multi' orqali yuklanadi
model_id = 'v5_cis_ext' # Siz tanlagan model ID
device = torch.device('cpu')

print(f"{model_id} yuklanmoqda, kuting...")

# Silero kutubxonasi modelni avtomatik topib yuklaydi
model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)

model.to(device)

# O'zbek tili uchun speaker tanlash
# Saida ovozi aynan shu model ichida mavjud
speaker = 'uzb_saida' 
sample_rate = 48000

print(f"Model yuklandi! Saqlangan joy: {torch.hub.get_dir()}")