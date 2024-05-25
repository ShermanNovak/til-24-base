import numpy as np
from torch.utils.data import Dataset
from helpers import convert_to_mel_spectrogram
from torch.nn.functional import pad
from scipy.io import wavfile
import torch
import librosa

TARGET_LENGTH = 3000
TARGET_NUM_SAMPLES = 480000

class AudioAugDataset(Dataset):
    def __init__(self, df, data_path, wav_augment=None, spec_augment=None, tokenizer=None):
        self.df = df
        self.tokenizer = tokenizer
        self.data_path = str(data_path)
        self.wav_augment = wav_augment
        self.spec_augment = spec_augment
        self.sample_rate = 16000
        self.channel = 1
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load audio file
        sample_rate, audio_data = wavfile.read(f"{self.data_path}/audio/{self.df.iloc[idx]['audio']}")
        
        # Convert audio data to floating-point format and numpy array
        audio_data = audio_data.astype(np.float32)
        
        # Pad the audio signal if it's shorter than the target duration
        if len(audio_data) < TARGET_NUM_SAMPLES:
            padding_needed = TARGET_NUM_SAMPLES - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_needed), mode='constant')
        
        if self.wav_augment:
            # Augment/transform/perturb the audio data
            audio_data = self.wav_augment(samples=audio_data, sample_rate=self.sample_rate)
        
        spectrogram = convert_to_mel_spectrogram(audio_data)

        if self.spec_augment:
            # Augment/transform/perturb the spectrogram
            spectrogram = self.spec_augment(spectrogram)
        
        # Convert to PyTorch tensor
        # audio_tensor = torch.from_numpy(audio_data)
        spectrogram_tensor = torch.from_numpy(spectrogram)
        
        # Pad spectrogram tensor to 3000
        target_length = TARGET_LENGTH
        if spectrogram_tensor.size(1) < target_length:
            padding = (0, target_length - spectrogram_tensor.size(1))
            spectrogram_tensor = pad(spectrogram_tensor, padding)
        
        # Get label from DataFrame
        label = self.df.iloc[idx]['transcript']
        
        input_ids = self.tokenizer(label).input_ids
        
        return {'input_features': spectrogram_tensor, 'label': input_ids}