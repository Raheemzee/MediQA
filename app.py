import tempfile
import subprocess
import io
import json
import os
import wave
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
from faster_whisper import WhisperModel

app = Flask(__name__)

# ===== Load QA dataset =====
files = [
    'medquad.csv', 'medsqud.csv', 'first_aid_dataset.csv'
]
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

questions = df['question'].fillna("").tolist()
answers = df['answer'].fillna("").tolist()

vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

# ===== Chatbot logic =====
def get_best_answer(user_question):
    user_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vec, question_vectors).flatten()
    best_idx = similarities.argmax()
    best_score = similarities[best_idx]
    return answers[best_idx] if best_score > 0.3 else "I'm not sure. Please consult a medical professional."

# ===== Text-to-Speech (gTTS) =====
def text_to_speech(text):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts = gTTS(text)
    tts.save(tmp_file.name)
    return tmp_file.name

# ===== Faster Whisper Model =====
model = WhisperModel("tiny", device="cpu")  # small/medium if Render plan has more RAM

@app.route('/stt', methods=['POST'])
def stt():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio_file = request.files['audio']

    # Save uploaded file
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    audio_file.save(tmp_input)

    # Convert WebM -> WAV (16kHz mono)
    tmp_wav = tmp_input.replace(".webm", ".wav")
    try:
        subprocess.run([
            "ffmpeg", "-i", tmp_input,
            "-ar", "16000", "-ac", "1", tmp_wav,
            "-y", "-loglevel", "quiet"
        ], check=True)
    except Exception as e:
        return jsonify({"error": f"ffmpeg conversion failed: {e}"}), 500

    # Transcribe with faster-whisper
    try:
        segments, _ = model.transcribe(tmp_wav)
        transcript = " ".join([seg.text for seg in segments]).strip()
    except Exception as e:
        return jsonify({"error": f"STT failed: {e}"}), 500

    # Get chatbot answer
    response = get_best_answer(transcript) if transcript else "Sorry, I didn’t catch that. Please try again."

    return jsonify({"transcript": transcript, "response": response})

# ===== Routes =====
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    chatbot_response = get_best_answer(user_input)
    return render_template('index.html', user_input=user_input, chatbot_response=chatbot_response)

@app.route('/speak_response', methods=['POST'])
def speak_response():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    audio_path = text_to_speech(text)
    return send_file(audio_path, mimetype="audio/mpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
