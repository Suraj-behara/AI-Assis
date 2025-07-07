
import openai
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import wavio
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

openai.api_key = "YOUR_OPENAI_API_KEY"
model = whisper.load_model("base")

def record_audio(duration=5, fs=16000):
    status_label.config(text="Recording...")
    root.update()
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    return audio, fs

def save_audio(audio, fs):
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(temp_wav.name, audio, fs, sampwidth=2)
    return temp_wav.name

def capture_and_respond():
    audio, fs = record_audio()
    file_path = save_audio(audio, fs)

    status_label.config(text="Transcribing...")
    root.update()
    result = model.transcribe(file_path)
    question = result["text"]
    text_box.insert(tk.END, f"\nYou asked: {question}\n")

    if question.strip() == "":
        status_label.config(text="Didn't catch that. Try again.")
        return

    status_label.config(text="Getting AI answer...")
    root.update()
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an AI interview assistant. Answer clearly and concisely."},
            {"role": "user", "content": question}
        ]
    )
    answer = response['choices'][0]['message']['content']
    text_box.insert(tk.END, f"AI Answer:\n{answer}\n")
    status_label.config(text="Ready")

root = tk.Tk()
root.title("AI Interview Assistant")

text_box = ScrolledText(root, wrap=tk.WORD, width=60, height=20)
text_box.pack(padx=10, pady=10)

record_button = tk.Button(root, text="Record Question", command=capture_and_respond)
record_button.pack(pady=5)

status_label = tk.Label(root, text="Ready")
status_label.pack(pady=5)

root.mainloop()
