import librosa
import pandas as pd
import json
import base64
import pickle

SAMPLE_RATE = 16000

def convert_to_mel_spectrogram(audio_data, sample_rate=SAMPLE_RATE):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=80)
    return mel_spectrogram

def read_data(file_path, dataframe_path):
    instances = []
    
    with open(f"{file_path}/asr.jsonl", "r") as f:
        for line in f:
            if line.strip() == "":
                continue
            instance = json.loads(line.strip())
            with open(f"{file_path}/audio/{instance['audio']}", "rb") as file:
                audio_bytes = file.read()
                instances.append(
                    {**instance, "b64": base64.b64encode(audio_bytes).decode("ascii")}
                )
                
    df = pd.DataFrame(instances)
    df.to_pickle(dataframe_path)

    return df

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}