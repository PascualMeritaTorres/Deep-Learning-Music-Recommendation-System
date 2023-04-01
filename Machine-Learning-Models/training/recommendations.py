# coding: utf-8
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import subprocess
import click

import torch
from torch.autograd import Variable

import model as Model
from paths import BINARY_PATH,TAGS_PATH,TEST_PATH,TRAIN_PATH,VALID_PATH,MODEL_LOAD_PATH,DATA_PATH,SAMPLE_SONG_PATH

class RetrieveSimilarSongs(object):
    """
    This class retrieves similar songs using the specified model.

    Attributes:
        config: A dictionary containing configuration parameters for the model.
    """
    def __init__(self, config):
        self.model_name = config.model_name
        self.model_load_path = config.model_load_path
        self.data_song_path = config.data_song_path
        self.batch_size = config.batch_size
        self.dataset_path=config.dataset_path
        self.fs = 16000

        self.binary_path=BINARY_PATH
        self.tags_path=TAGS_PATH
        self.test_path=TEST_PATH
        self.train_path=TRAIN_PATH
        self.validate_path=VALID_PATH

        self.get_cvs()
        self.build_model()

        if os.path.isfile(self.data_song_path) and self.data_song_path.endswith('.wav'):
            print(f"Playing your input song")
            process = subprocess.Popen(['afplay', self.data_song_path])
            input("Press Enter to stop playback and move to recommended songs...")
            process.terminate()
        else:
            print(f"{self.data_song_path} is not a valid WAV audio file")
        
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
        self.train_list = np.load(self.train_path)
        self.test_list = np.load(self.test_path)
        self.valid_list = np.load(self.validate_path)
        self.all_songs_list = np.concatenate([self.train_list, self.test_list, self.valid_list], axis=0)
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
        raw, sr = librosa.core.load(self.data_song_path, sr=self.fs)
        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, 1, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x


    def get_single_audio_loader(self):
        """
        Get the single audio loader by loading the audio.

        Returns:
            torch.Tensor: A tensor containing the loaded audio data.
        """
        audio = self.load_audio()
        return audio
    
    def plot_tag_graph(self,sortedList):
        """
        Plot a bar chart of the top 5 tags and their confidence values.

        Args:
            sortedList (list): A list of tuples containing the tags and their confidence values.
        """
        # Extract the top 15 tags and values
        top_tags = [x[0] for x in sortedList[:5]]
        top_values = [x[1] for x in sortedList[:5]]
        # Create a bar chart
        plt.bar(top_tags, top_values)
        plt.ylim(0.0, 1.0)  # Set y-axis limits
        plt.title('Top 5 tags')
        plt.xlabel('')
        plt.ylabel('Confidence')
        plt.show()


    def get_top_tags(self):
        """
        Get the top tags for the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        self.model = self.model.eval()
        data_loader = self.get_single_audio_loader()
        audio_tensor = next(iter(data_loader))
        x = Variable(audio_tensor)
        out = self.model(x)
        out = out.detach().cpu()
        out = out.detach().numpy().reshape(-1)
        pairs = []
        for i in range(len(self.tags)):
            pairs.append((self.tags[i], out[i]))
        sortedList = sorted(pairs, key=lambda x: -x[1])
        return sortedList
    
    def find_similar_songs(self, sortedList):
        """
        Find and play similar songs based on the top tags.

        Args:
            sortedList (list): A list of tuples containing the top tags and their confidence values.
        """
        originalIndices = [sortedList.index(x) for x in sortedList[:5]]
        relevantColumnsOfAllSongs = self.binary[:, originalIndices]
        rowSums = relevantColumnsOfAllSongs.sum(axis=1)
        sorted_indices = np.argsort(rowSums)[::-1]

        max_sum_rows = []
        for i in sorted_indices:
            if len(max_sum_rows) >= 5:
                break
            if not max_sum_rows or rowSums[i] == rowSums[max_sum_rows[-1]]:
                max_sum_rows.append(i)
            else:
                break

        top_rows = np.random.choice(max_sum_rows, size=min(5, len(max_sum_rows)), replace=False)

        arr_split = np.array([s.split('\t') for s in self.all_songs_list])
        mask = np.in1d(arr_split[:, 0].astype(int), top_rows)
        file_names = arr_split[mask][:, 1]

        counter = 1
        for song_path in file_names:
            full_path = os.path.join(self.dataset_path, song_path)
            if os.path.isfile(full_path) and full_path.endswith('.mp3'):
                print('Playing recommendation', counter)
                process = subprocess.Popen(['afplay', full_path])
                counter += 1
                input("Press Enter to stop playback and move to next song...")
                process.terminate()
            else:
                print(f"{full_path} is not a valid MP3 audio file")
        
    def give_song_recommendations(self):
        """
        Give song recommendations based on the input song.

        Returns:
            list: A list of tuples containing the top tags and their confidence values.
        """
        sortedList = self.get_top_tags()
        self.plot_tag_graph(sortedList)
        self.find_similar_songs(sortedList)
        return sortedList


@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='fcn', help='Model type to use')
@click.option('--batch_size', type=int, default=16, help='Number of samples passed through to the network at one time')
@click.option('--model_load_path', type=str, default=MODEL_LOAD_PATH, help='Path to load the saved model')
@click.option('--data_song_path', type=str, default=SAMPLE_SONG_PATH, help='Path to the test song')
@click.option('--dataset_path', type=str, default=DATA_PATH, help='Path to the dataset')
def run(model_name, batch_size, model_load_path, data_song_path, dataset_path):
    """
    This script retrieves similar songs using the specified model.

    Args:
        model_type: Model type to use
        batch_size: Number of samples passed through to the network at one time
        model_load_path: Path to load the saved model
        data_song_path: Path to the test song
        dataset_path: Path to the dataset
    """
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name=model_name,
        batch_size=batch_size,
        model_load_path=model_load_path,
        data_song_path=data_song_path,
        dataset_path=dataset_path
    )

    print(config)
    s = RetrieveSimilarSongs(config)
    recommendations = s.give_song_recommendations()
    print(recommendations)

if __name__ == '__main__':
    run()
