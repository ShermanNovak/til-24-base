import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class ASRManager:
    def __init__(self):
        model_id = "distil-whisper/distil-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, low_cpu_mem_usage=True, use_safetensors=True
        )

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
        )

        
    def transcribe(self, audio_bytes: bytes) -> str:
    
        # perform ASR transcription
        with torch.cuda.amp.autocast():
            text = self.pipe(audio_bytes)["text"]
    
        return text

