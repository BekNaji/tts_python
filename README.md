# Silero TTS API Server

A lightweight Flask API for text-to-speech conversion using **Silero TTS** models.

## Features

* **Languages:** Uzbek (`uz`), Russian (`ru`), and English (`en`).
* **Auto-Download:** Fetches missing `.pt` model files automatically on startup.
* **Text Processing:** Automatic number-to-word normalization and long text chunking.
* **Customization:** Adjustable speech `speed` and `speaker` selection.

## Requirements
* **Python Version:** `3.11.x`
* **OS:** Windows, macOS, or Linux

## Installation (using venv)

1. **Clone and Enter:**
```bash
git clone <repository_url>
cd tts_python

```


2. **Virtual Environment:**
```bash
python -m venv venv
# Activate (Windows)
.\venv\Scripts\Activate.ps1
# Activate (Linux/Mac)
source venv/bin/activate

```


3. **Install & Run:**
```bash
pip install -r requirements.txt
python run.py

```



## API Usage

**Endpoint:** `POST /tts`

**Request Example:**

```json
{
  "text": "Hello world",
  "lang": "en",
  "speaker": "en_0",
  "file": "./audio.waw"
}

```

**Response:**

```json
{
  "status": "success",
  "file": "output.wav"
}

```

---
