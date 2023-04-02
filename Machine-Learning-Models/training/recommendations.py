# coding: utf-8
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import subprocess
import click
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


import torch
from torch.autograd import Variable

import model as Model
from paths import BINARY_PATH,TAGS_PATH,MODEL_LOAD_PATH,DATA_PATH,SAMPLE_SONG_PATH,FULL_DATASET_PATH,DATA_PATH

class RetrieveSimilarSongs(object):
    """
    This class retrieves similar songs using the specified model.

    Attributes:
        config: A dictionary containing configuration parameters for the model.
    """
    def __init__(self, config):
        self.model_name = config.model_name
        self.model_load_path = config.model_load_path
        self.sample_song_path = config.sample_song_path
        self.batch_size = config.batch_size
        self.songs_path=config.songs_path
        self.fs = 16000

        self.full_dataset_path=FULL_DATASET_PATH
        self.binary_path=BINARY_PATH
        self.tags_path=TAGS_PATH
        self.data_path=DATA_PATH

        self.get_cvs()
        self.build_model()

        if os.path.isfile(self.sample_song_path) and self.sample_song_path.endswith('.wav'):
            print(f"Playing your input song")
            process = subprocess.Popen(['afplay', self.sample_song_path])
            input("Press Enter to stop playback and move to recommended songs...")
            process.terminate()
        else:
            print(f"{self.sample_song_path} is not a valid WAV audio file")
        
    def build_model(self):
        """
        Build the model by getting the model and loading the model from the specified path.
        """
        if self.model_name == 'fcn':
            self.input_length = 29 * 16000
            self.model=Model.FCN()
        elif self.model_name == 'crnn':
            self.input_length = 29 * 16000
            self.model=Model.CRNN()
        elif self.model_name == 'short':
            self.input_length = 59049
            self.model=Model.ShortChunkCNN()
        elif self.model_name == 'short_res':
            self.input_length = 59049
            self.model=Model.ShortChunkCNN_Res()

        # load model
        self.load_model(self.model_load_path)


    def get_cvs(self):
        """
        Load the train, test, and validation lists, binary data, and tags data from the specified paths.
        """
        self.full_dataset=pd.read_csv(self.full_dataset_path)
        self.binary = np.load(self.binary_path)
        self.tags = np.load(self.tags_path)

    def load_model(self, filename):
        """
        Load the model from the specified filename.

        Args:
            filename (str): The path to the file containing the saved model.
        """
        S = torch.load(filename, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def load_audio(self):
        """
        Load the audio data from the specified data song path.

        Returns:
            torch.Tensor: A tensor containing the loaded audio data
        """
        numpy_song, sr = librosa.core.load(self.sample_song_path, sr=self.fs)
        random_idx = int(np.floor(np.random.random(1) * (len(numpy_song)-self.input_length))) #generate a random integer to select a random starting point in the numpy array
        sliced_numpy_song = np.array(numpy_song[random_idx:random_idx+self.input_length]) #slice the original .npy array, starting at the random index and ending at the random index+input_length
        sliced_numpy_song=sliced_numpy_song.astype('float32')

        # split chunk
        length = len(sliced_numpy_song)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(sliced_numpy_song[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x
    


    def get_features(self):
        """
        Get the top tags for the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.model = self.model.eval()
        audio_tensor = self.load_audio()
        x = Variable(audio_tensor)
        out = self.model(x)
        out = out.detach().cpu()
        mean_out = np.array(out).mean(axis=0)
        return mean_out
    
    def find_similar_songs(self):
        """
        Find and play similar songs based on the top tags.

        Args:
            sortedList (list): A list of tuples containing the top tags and their confidence values.
        """
        feature_columns = ['danceability', 'energy', 'loudness', 'mode', 'acousticness', 'instrumentalness',
                           'liveness', 'valence', '78-92bpm', '92-101bpm', '101-110bpm', '110-120bpm', '120-128bpm',
                           '128-140bpm', '140-154bpm', '154-170bpm', 'A#_Bb-Key', 'A-Key', 'B-Key', 'C#_Db-Key', 'C-Key',
                           'D#_Eb-Key', 'D-Key', 'E-Key', 'F#_Gb-Key', 'F-Key', 'G#_Ab-Key', 'G-Key', '1-time-sign',
                           '3-time-sign', '4-time-sign', '5-time-sign']
        input_song_features=self.get_features()
        input_song_features = input_song_features.reshape(1, -1)

        dataset_features = self.full_dataset[feature_columns].values

        # Compute cosine similarity between the input song and all songs in the dataset
        similarities = cosine_similarity(input_song_features, dataset_features)

        # Find the indices of the top 5 most similar songs
        top_indices = np.argsort(similarities[0])[-5:][::-1]

        # Play the top 5 similar songs
        for idx in top_indices:
            song = self.full_dataset.iloc[idx]
            print(f"Playing {song['track_name']} by {song['artist_name']} (similarity: {similarities[0][idx]:.4f})")
            full_path = os.path.join(self.data_path, song['track_uri']+'.mp3')
            print("full path",full_path)
            process = subprocess.Popen(['afplay', full_path])
            input("Press Enter to stop playback and move to the next recommended song...")
            process.terminate()

        return top_indices
        
        
        
    def give_song_recommendations(self):
        """
        Give song recommendations based on the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        return self.find_similar_songs()

@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='short_res', help='Model type to use')
@click.option('--batch_size', type=int, default=16, help='Number of samples passed through to the network at one time')
@click.option('--model_load_path', type=str, default=MODEL_LOAD_PATH, help='Path to load the saved model')
@click.option('--sample_song_path', type=str, default=SAMPLE_SONG_PATH, help='Path to the test song')
@click.option('--songs_path', type=str, default=DATA_PATH, help='Path to the songs dataset')
def run(model_name, batch_size, model_load_path, sample_song_path, songs_path):
    """
    This script retrieves similar songs using the specified model.

    Args:
        model_type: Model type to use
        batch_size: Number of samples passed through to the network at one time
        model_load_path: Path to load the saved model
        sample_song_path: Path to the test song
        songs_path: Path to the song dataset
    """
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name=model_name,
        batch_size=batch_size,
        model_load_path=model_load_path,
        sample_song_path=sample_song_path,
        songs_path=songs_path
    )

    print(config)
    s = RetrieveSimilarSongs(config)
    recommendations = s.give_song_recommendations()
    print(recommendations)

if __name__ == '__main__':
    run()
