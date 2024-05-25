import evaluate
import os
import torch
import pandas as pd
import numpy as np

from DataCollator import DataCollator
from AudioAugDataset import AudioAugDataset

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, SpecCompose, SpecFrequencyMask
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from helpers import read_data, compute_metrics
from torch.utils.data import random_split

wav_augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
])

spec_augment = SpecCompose(
    [
        SpecFrequencyMask(p=0.5),
    ]
)

DATAFRAME_PATH = '../tmp/asr_dataframe.pkl'
FILE_PATH = "../../../advanced"

if os.path.exists(DATAFRAME_PATH):
    df = pd.read_pickle(DATAFRAME_PATH) 
else:
    df = read_data(FILE_PATH, DATAFRAME_PATH)

print(df.head())

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.generation_config.language = "english"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# from faster_whisper import WhisperModel

# model = WhisperModel("distil-large-v2")

# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")

audioAugDataset = AudioAugDataset(df, FILE_PATH, wav_augment, spec_augment, tokenizer)

num_items = len(audioAugDataset)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(audioAugDataset, [num_train, num_val])

print(train_ds[0])
print(train_ds[1]['input_features'].shape)

metric = evaluate.load("wer")

data_collator = DataCollator(
    tokenizer=tokenizer,
    feature_extractor=feature_extractor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi",  # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    # fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=25,
    eval_steps=25,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# trainer.train(resume_from_checkpoint=True)
trainer.train()

kwargs = {
    "dataset_tags": "DSTA BrainHack TILAI",
    "dataset": "DSTA BrainHack TILAI",  # a 'pretty' name for the training dataset
    "dataset_args": "config: en, split: test",
    "language": "en",
    "model_name": "Whisper BrainHack TILAI",  # a 'pretty' name for your model
    "finetuned_from": "openai/whisper-small",
    "tasks": "automatic-speech-recognition",
}

trainer.push_to_hub(**kwargs)