# coding: utf-8
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import click
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


import torch
from torch.autograd import Variable

import model as Model
from paths import BINARY_PATH,TAGS_PATH,MODEL_LOAD_PATH,DATA_PATH,SAMPLE_SONG_PATH,FULL_DATASET_PATH,DATA_PATH,MTAT_TAGS_PATH

class RetrieveSimilarSongs(object):
    """
    This class retrieves similar songs using the specified model.

    Attributes:
        config: A dictionary containing configuration parameters for the model.
    """
    def __init__(self, config):
        self.short_res_model_load_path = config.short_res_model_load_path
        self.crnn_model_load_path = config.crnn_model_load_path
        self.sample_song_path = config.sample_song_path
        self.songs_path=config.songs_path
        self.fs = 16000
        self.batch_size=config.batch_size

        self.spotify_csv_dataset_path=FULL_DATASET_PATH
        self.tags_dataset_path='./tagged_songs.csv'
        self.tags_path=MTAT_TAGS_PATH

        self.get_cvs()
        self.build_model()

        
    def build_model(self):
        """
        Build the model by getting the model and loading the model from the specified path.
        """
        self.crnn_input_length = 29 * 16000
        self.crnn_model=Model.CRNNSOTA().to(torch.device("cpu"))
        self.short_res_input_length = 59049
        self.short_res_model=Model.ShortChunkCNN_Res().to(torch.device("cpu"))

        # load model
        self.load_short_res_model(self.short_res_model_load_path)
        self.load_crnn_model(self.crnn_model_load_path)


    def get_cvs(self):
        """
        Load the train, test, and validation lists, binary data, and tags data from the specified paths.
        """
        self.spotify_csv_dataset=pd.read_csv(self.spotify_csv_dataset_path)
        self.tags_dataset=pd.read_csv(self.tags_dataset_path)
        self.tags=np.load(self.tags_path)

    def load_crnn_model(self, filename):
        """
        Load the model from the specified filename.

        Args:
            filename (str): The path to the file containing the saved model.
        """
        S = torch.load(filename, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.crnn_model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.crnn_model.load_state_dict(S)

    def load_short_res_model(self, filename):
        """
        Load the model from the specified filename.

        Args:
            filename (str): The path to the file containing the saved model.
        """
        S = torch.load(filename, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.short_res_model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.short_res_model.load_state_dict(S)

    def load_audio(self,input_length):
        """
        Load the audio data from the specified data song path.

        Returns:
            torch.Tensor: A tensor containing the loaded audio data
        """
        signal, sr = librosa.core.load(self.sample_song_path, sr=self.fs)
        length = len(signal)
        hop = length // 2 - input_length // 2
        x = torch.zeros(1, input_length)
        x[0] = torch.Tensor(signal[hop : hop + input_length]).unsqueeze(0)
        x = Variable(x.to(torch.device("cpu")))
        return x
    


    def get_spotify_features(self):
        """
        Get the top tags for the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.short_res_model = self.short_res_model.eval()
        x = self.load_audio(self.short_res_input_length)
        out = self.short_res_model(x)
        out= out[0].detach().numpy().tolist()
        return [out]

    def get_tags(self):
        """
        Get the top tags for the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.crnn_model = self.crnn_model.eval()
        x = self.load_audio(self.crnn_input_length)
        out = self.crnn_model(x)
        out= out[0].detach().numpy().tolist()
        return out
    
    def find_similar_songs_spotify_features(self,recommended_songs):
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
        input_song_features=self.get_spotify_features()

        # Filter the dataset_features by only considering the recommended songs
        recommended_uris = [song['track_uri'] for song in recommended_songs]
        filtered_dataset = self.spotify_csv_dataset[self.spotify_csv_dataset['track_uri'].isin(recommended_uris)]


        dataset_features = filtered_dataset[feature_columns].values

        # Compute cosine similarity between the input song and all songs in the dataset
        similarities = cosine_similarity(input_song_features, dataset_features)

        # Find the indices of the top 5 most similar songs
        top_indices = np.argsort(similarities[0])[-5:][::-1]

       # Return the top 5 similar songs as a list of dictionaries
        similar_songs = []
        for idx in top_indices:
            song = self.spotify_csv_dataset.iloc[idx]
            similarity = similarities[0][idx]
            similar_songs.append({
                'track_name': song['track_name'],
                'artist_name': song['artist_name'],
                'similarity': round(similarity, 4),
                'song_id': song['track_uri']+'.mp3' 
            })

        return similar_songs
    
    def plot_tag_graph(self,sortedList):
        """
        Plot a bar chart of the top 10 tags and their confidence values.
        Args:
            sortedList (list): A list of tuples containing the tags and their confidence values.
        """
        # Extract the top 15 tags and values
        top_tags = [x[0] for x in sortedList[:10]]
        top_values = [x[1] for x in sortedList[:10]]
        # Create a bar chart
        plt.bar(top_tags, top_values)
        plt.ylim(0.0, 1.0)  # Set y-axis limits
        plt.title('Top 10 tags')
        plt.xlabel('')
        plt.ylabel('Confidence')
        plt.savefig("top_tags.png")

    def get_top_tags(self):
        """
        Get the top tags for the input song.
        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.crnn_model = self.crnn_model.eval()
        x = self.load_audio(self.crnn_input_length)
        out = self.crnn_model(x)
        out= out[0].detach().numpy().tolist()
        pairs = []
        print('The size of the tags array is ' + str(len(self.tags)))

        for i in range(len(self.tags)):
            pairs.append((self.tags[i], out[i]))
        sortedList = sorted(pairs, key=lambda x: -x[1])
        return sortedList
    
    def find_similar_songs_tags(self):
        wanted_tags = ['guitar','classical','slow','techno','strings','drums','electronic','rock','fast',
                'piano','ambient','beat','violin','vocal','synth','indian','opera','vocals','no vocals','pop','classic','dance','cello']
        
        all_tags = ['guitar','classical','slow','techno','strings','drums','electronic','rock','fast',
                'piano','ambient','beat','violin','vocal','synth','female','indian','opera','male',
                'singing','vocals','no vocals','harpsichord','loud','quiet','flute','woman','male vocal',
                'no vocal','pop','soft','sitar','solo','man','classic','choir','voice','new age','dance',
                'male voice','female vocal','beats','harp','cello','no voice','weird','country','metal',
                'female voice','choral']
        
        indexes = [all_tags.index(tag) for tag in wanted_tags if tag in all_tags]

        # Get only the tags from get_tags() that are at those indexes:
        input_song_tags = self.get_tags()
        input_song_tags = [input_song_tags[i] for i in indexes]

        # Read the dataset
        dataset = self.tags_dataset
        # Convert values to binary based on the given condition
        dataset[wanted_tags] = dataset[wanted_tags].applymap(lambda x: 1 if x > 0.3 else 0)

        # Calculate the number of common tags
        dataset['common_tags'] = (dataset[wanted_tags].values == input_song_tags).sum(axis=1)

        # Filter rows with at least 2 common tags
        filtered_dataset = dataset[dataset['common_tags'] >= 1]

        # Find track_uris with the most common tags
        max_common_tags = filtered_dataset['common_tags'].max()
        most_common_tag_tracks = filtered_dataset[filtered_dataset['common_tags'] == max_common_tags]

        # Create a list of dictionaries containing track_uri for each similar song
        similar_songs = []
        for idx, row in most_common_tag_tracks.iterrows():
            similar_songs.append({
                'track_uri': row['track_uri'],
            })

        return similar_songs


        
    def give_song_recommendations(self):
        """
        Give song recommendations based on the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.plot_tag_graph(self.get_top_tags())

        first_filter_recommended_songs=self.find_similar_songs_tags()
        print(first_filter_recommended_songs)
        recommended_songs=self.find_similar_songs_spotify_features(first_filter_recommended_songs)
        return recommended_songs


@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='short_res', help='Model type to use')
@click.option('--batch_size', type=int, default=16, help='Number of samples passed through to the network at one time')
@click.option('--short_res_model_load_path', type=str, default=MODEL_LOAD_PATH, help='Path to load the short_res saved model')
@click.option('--crnn_model_load_path', type=str, default=MODEL_LOAD_PATH, help='Path to load the crnn saved model')
@click.option('--sample_song_path', type=str, default=SAMPLE_SONG_PATH, help='Path to the test song')
@click.option('--songs_path', type=str, default=DATA_PATH, help='Path to the songs dataset')
def run(model_name, batch_size,short_res_model_load_path,crnn_model_load_path, sample_song_path, songs_path):
    """
    This script retrieves similar songs using the specified model.

    Args:
        model_type: Model type to use
        batch_size: Number of samples passed through to the network at one time
        short_res_model_load_path: Path to load the saved model
        crnn_model_load_path: Path to load the saved model
        sample_song_path: Path to the test song
        songs_path: Path to the song dataset
    """
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name=model_name,
        batch_size=batch_size,
        short_res_model_load_path=short_res_model_load_path,
        crnn_model_load_path=crnn_model_load_path,
        sample_song_path=sample_song_path,
        songs_path=songs_path
    )

    print(config)
    s = RetrieveSimilarSongs(config)
    recommendations = s.give_song_recommendations()
    print(recommendations)

if __name__ == '__main__':
    run()
