import pandas as pd
import os
import torch
from helpers import read_data
from datasets import Audio, Dataset, DatasetDict
from dataclasses import dataclass
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any, Dict, List, Union
import evaluate
import pickle
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
import base64

DATASETDICT_PATH = './tmp/asr_datasetdict.pkl'
DATAFRAME_PATH = './tmp/asr_dataframe.pkl'
FILE_PATH = "../../advanced"
DIRECTORY_PREFIX = "../../advanced/audio/"

model_name_or_path = "openai/whisper-base.en"
task = "transcribe"
language = "English"
language_abbr = "en" # Short hand code for the language we want to fine-tune
peft_model_id = "faster-peft"

# Assuming df is your DataFrame
def df_to_dataset(df):
    df['audio'] = df['audio'].apply(lambda x: DIRECTORY_PREFIX + x)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    return dataset


def prepare_dataset(batch):
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language_abbr, task=task)
    
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch



def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

    

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


def get_dataset_dict():
    if os.path.exists(DATASETDICT_PATH):
        with open(DATASETDICT_PATH, "rb") as f:
            dataset_dict = pickle.load(f)
    else:
        if os.path.exists(DATAFRAME_PATH):
            df = pd.read_pickle(DATAFRAME_PATH) 
        else:
            df = read_data(FILE_PATH, DATAFRAME_PATH)

        print(df.head())

        # Split the DataFrame into train, test, and validation DataFrames
        train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
        test_df = df.drop(train_df.index)

        # Convert DataFrames to Datasets
        train_dataset = df_to_dataset(train_df)
        test_dataset = df_to_dataset(test_df)

        # Create a DatasetDict
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })

        dataset_dict = dataset_dict.remove_columns(
            ["__index_level_0__"]
        )
        
        dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=1)
            
        with open(DATASETDICT_PATH, "wb") as f:
            pickle.dump(dataset_dict, f)
            
        print(dataset_dict)
        print(dataset_dict['train'][0])
    
    return dataset_dict


if __name__ == '__main__':
# def train():
    processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
    
    config = WhisperConfig.from_pretrained(model_name_or_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True))
    
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, config=config)
    model.config.max_length=150
    
    dataset_dict = get_dataset_dict()
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")
    
    model = prepare_model_for_kbit_training(model)
    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    
    config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./helps-peft",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-3,
        warmup_steps=50,
        num_train_epochs=1,
        evaluation_strategy="steps",
        gradient_checkpointing=True,
        fp16=True,
        per_device_eval_batch_size=8,
        generation_max_length=128,
        save_steps=100,
        logging_steps=25,
        # eval_steps=5,
        # max_steps=5, # only for testing purposes, remove this from your final run :)
        remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
        label_names=["labels"],  # same reason as above
        push_to_hub=True,
        load_best_model_at_end=True,
    )
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[SavePeftModelCallback],
    )
    model.config.use_cache = False 
    
    trainer.train()
    
    model.config.to_json_file('adapter_config.json')
    model.push_to_hub(peft_model_id)
