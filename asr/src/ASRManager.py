import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class ASRManager:
    def __init__(self):
        self.pipe = pipeline(
          "automatic-speech-recognition",
          model="openai/whisper-base.en",
          chunk_length_s=30,
          device=device,
        )
        
    def transcribe(self, audio_bytes: bytes) -> str:
    
        # perform ASR transcription
        with torch.cuda.amp.autocast():
            text = self.pipe(audio_bytes)["text"]
    
        return text

