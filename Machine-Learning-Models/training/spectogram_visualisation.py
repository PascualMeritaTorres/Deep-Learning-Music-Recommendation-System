import numpy as np
import torchaudio
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class AudioTransformations:
    def __init__(self, sample_rate, n_fft, f_min, f_max, n_mels):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            f_min=self.f_min,
            f_max=self.f_max,
            n_mels=self.n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def process_audio(self, file_path, duration, save_path):
        # Load the numpy array
        data = np.load(file_path, mmap_mode='r') 

        # Extract the first n seconds
        samples = self.sample_rate * duration
        data_n_seconds = data[:samples]

        # Create a torch tensor
        data_tensor = torch.from_numpy(data_n_seconds.copy()).float()

        # Perform the transformations
        mel_spec = self.mel_spectrogram(data_tensor.unsqueeze(0))
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Save the diagrams
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))

        axes[0].imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='jet')
        axes[0].set_title('Mel Spectrogram')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mel Frequency')

        axes[1].imshow(mel_spec_db.squeeze().numpy(), aspect='auto', origin='lower', cmap='jet')
        axes[1].set_title('Amplitude to dB')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mel Frequency')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

if __name__ == "__main__":
    sample_rate = 16000
    n_fft = 512
    f_min = 0.0
    f_max = 8000.0
    n_mels = 128

    audio_transformer = AudioTransformations(sample_rate, n_fft, f_min, f_max, n_mels)
    #The song is Viva la Vida by Coldplay
    audio_transformer.process_audio('/Users/pascualmeritatorres/Developer/Dissertation/actual-dissertation/Dataset-Creation-And-Preprocessing/notebooks/mp3-new-download-session/npy/1mea3bSkSGXuIRvnydlB5b.npy', 20, 'transformations.png')
