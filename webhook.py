import os
import requests
import tempfile
from flask import Flask, request, jsonify

# --------- CONFIG ---------
LLM_SERVER_URL = "http://your-llm-server.com/process"  # <-- เปลี่ยนเป็น server ของพี่ตูน
STT_API_URL = "https://api.openai.com/v1/audio/transcriptions"
TTS_API_URL = "https://api.openai.com/v1/audio/speech"
OPENAI_API_KEY="key in discord or others"

# --------- APP ---------
app = Flask(__name__)

def speech_to_text(audio_file_path):
    with open(audio_file_path, "rb") as f:
        response = requests.post(
            STT_API_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": f},
            data={"model": "whisper-1"}
        )
    return response.json().get("text", "")

def split_into_verses(text: str, max_len: int = 20):
    """
    แบ่งข้อความเป็น verse (ทีละประโยคสั้นๆ)
    """
    if not text.strip():
        return []
    
    words = text.split()
    verses, current = [], []
    
    for w in words:
        current.append(w)
        if len(current) >= max_len or w.endswith(('.', '!', '?', '।', '๏')):
            verses.append(" ".join(current))
            current = []
    
    if current:
        verses.append(" ".join(current))
    
    return verses

def text_to_speech(text):
    response = requests.post(
        TTS_API_URL,
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={"model": "gpt-4o-mini-tts", "voice": "alloy", "input": text}
    )
    return response.content  # binary audio data

def forward_to_llm(text: str, audio_bytes: bytes):
    """
    ส่ง text + audio ไปให้ LLM server ของทีมอื่น
    """
    files = {"audio": ("output.wav", audio_bytes, "audio/wav")}
    data = {"text": text}
    try:
        response = requests.post(LLM_SERVER_URL, data=data, files=files, timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Webhook หลัก: 
    - รับไฟล์เสียงจาก LINE OA
    - STT → Verse → TTS → ส่งต่อให้ LLM
    """
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No audio file received"}), 400

    # เก็บไฟล์ temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name

    # STT
    text = speech_to_text(audio_path)

    # แบ่ง verse
    verses = split_into_verses(text)

    # TTS (รวมทุก verse กลับมาเป็นเสียงสั้นๆ <= 1 นาที)
    audio_out = b""
    for v in verses:
        audio_out += text_to_speech(v)

    # ส่งต่อไปยัง LLM server
    result = forward_to_llm(text, audio_out)

    # ส่ง response กลับ (debug เท่านั้น)
    return jsonify({
        "received_text": text,
        "verses": verses,
        "llm_response": result
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
