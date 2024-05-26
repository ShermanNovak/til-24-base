from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperConfig, 
    BitsAndBytesConfig
)
import torch
from peft import PeftModel, PeftConfig

class ASRManager:
    def __init__(self):
        peft_model_id = "ShermanNovak/helps-peft" 
        language = "en"
        task = "transcribe"

        peft_config = PeftConfig.from_pretrained(peft_model_id)

        config = WhisperConfig.from_pretrained(peft_config.base_model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True))

        model = WhisperForConditionalGeneration.from_pretrained(
            peft_config.base_model_name_or_path, config=config
        )
        model = PeftModel.from_pretrained(model, peft_model_id)
        model.config.use_cache = True

        tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
        feature_extractor = processor.feature_extractor
        self.forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        self.pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor)

        
    def transcribe(self, audio_bytes: bytes) -> str:
        # Start the timer
        # start_time = time.time()
    
        # perform ASR transcription
        with torch.cuda.amp.autocast():
            text = self.pipe(audio_bytes, generate_kwargs={"forced_decoder_ids": self.forced_decoder_ids}, max_new_tokens=255)["text"]
            
        # Stop the timer
        # end_time = time.time()
        
        # Calculate the duration
        # duration = end_time - start_time
        # print(f"Transcription took {duration:.2f} seconds")
    
        return text

