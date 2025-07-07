
import openai
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import wavio

openai.api_key = "YOUR_OPENAI_API_KEY"
model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio, fs

def save_audio(audio, fs):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_wav.name, audio, fs, sampwidth=2)
    return temp_wav.name

while True:
    audio, fs = record_audio()
    file_path = save_audio(audio, fs)

    print("Transcribing...")
    result = model.transcribe(file_path)
    question = result["text"]
    print(f"You asked: {question}")

    if question.strip() == "":
        print("Didn't catch that. Try again.")
        continue

    print("Getting AI answer...")

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI interview assistant. Answer clearly and concisely."},
            {"role": "user", "content": question}
        ]
    )

    answer = response['choices'][0]['message']['content']
    print(f"\nAI Answer:\n{answer}\n")
