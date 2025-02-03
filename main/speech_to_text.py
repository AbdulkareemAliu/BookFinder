import sys
import tty
import json
import termios
import threading
from queue import Queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class SpeechHandler:
    def __init__(self, model_path, start_stop_key=" ", exit_key="c"):
        assert start_stop_key != exit_key, "Start/stop key must differ from exit key"
        # download model from https://alphacephei.com/vosk/models
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.audio_queue = Queue()
        self.listening = False
        self.transcribed_text = []

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio error: {status}")
        self.audio_queue.put(bytes(indata))

    def listen_and_transcribe(self):
        with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype='int16',
                               channels=1, callback=self.audio_callback, latency='low'):

            print("Started listening")
            while self.listening:
                data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    self.transcribed_text.append(result.get("text", ""))
                    
                    print("Recorded:", result.get("text", "").strip())

    def toggle_listening(self):
        if not self.listening:
            self.listening = True
            threading.Thread(target=self.listen_and_transcribe, daemon=True).start()
        else:
            print("Stopped listening")
            self.listening = False
            if self.transcribed_text:
                text = " ".join(self.transcribed_text)
                self.transcribed_text.clear()
                return text

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def record(self, key_to_press=" "):
        stop_listening = False
        while not stop_listening:
            key = self.get_key()
            if key == key_to_press and not self.listening:
                self.toggle_listening()
            elif key == key_to_press and self.listening:
                transcribed_text = self.toggle_listening()
                stop_listening = True

        return transcribed_text

if __name__ == "__main__":
    handler = SpeechHandler("../models/vosk-model-small-en-us-0.15")

    print(sd.query_devices())
    transcription = handler.record("p")
    print(transcription)