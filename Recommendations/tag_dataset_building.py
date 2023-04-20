import os
import numpy as np
import pandas as pd
import torch
import librosa
from torch.autograd import Variable
from paths import CRNN_MODEL_PATH,MTAT_TAGS_PATH,FULL_DATASET_PATH,DATA_NUMPY_PATH
from ..Short_ChunkCNNRes_Model.training import model as Model
from tqdm import tqdm

class MusicTagger:
    def __init__(self, full_dataset_path, numpy_songs_path, model_name, fs, batch_size):
        self.full_dataset_path = full_dataset_path
        self.numpy_songs_path = numpy_songs_path
        self.model_name = model_name
        self.fs = fs
        self.batch_size = batch_size

        self.tags_mtat_path = MTAT_TAGS_PATH

        self.get_cvs()
        self.build_model()

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
        self.load_model(CRNN_MODEL_PATH)


    def get_cvs(self):
        """
        Load the train, test, and validation lists, binary data, and tags data from the specified paths.
        """
        self.full_dataset=pd.read_csv(self.full_dataset_path)

    def load_model(self, filename):
        """
        Load the model from the specified filename.

        Args:
            filename (str): The path to the file containing the saved model.
        """
        S = torch.load(filename, map_location=torch.device('cpu'))
        
        # Filter the state dictionary to match the current model's keys
        model_dict = self.model.state_dict()
        filtered_dict = {k: v for k, v in S.items() if k in model_dict}
        model_dict.update(filtered_dict)

        self.model.load_state_dict(model_dict)


    def make_tags_csv(self):
        # Create an empty DataFrame with the required columns
        tagged_songs = pd.DataFrame(columns=['track_uri'] + np.load(self.tags_mtat_path).tolist())

        # Iterate through each row in the full dataset
        for index, row in tqdm(self.full_dataset.iterrows(), total=self.full_dataset.shape[0]):
            # Find the mp3 in MP3_DATA_PATH by getting the element in the folder that is called row['track_uri']+'.mp3'
            song_path = os.path.join(self.numpy_songs_path, row['track_uri'] + '.npy')
            # Get the features for the current song
            self.model = self.model.eval()
            numpy_song = np.load(song_path, mmap_mode='r') #load the npy file into memory
            random_idx = int(np.floor(np.random.random(1) * (len(numpy_song)-self.input_length))) #generate a random integer to select a random starting point in the numpy array
            sliced_numpy_song = np.array(numpy_song[random_idx:random_idx+self.input_length]) #slice the original .npy array, starting at the random index and ending at the random index+input_length
            sliced_numpy_song=sliced_numpy_song.astype('float32')
            # split chunk
            length = len(sliced_numpy_song)
            hop = (length - self.input_length) // self.batch_size
            x = torch.zeros(self.batch_size, self.input_length)
            for i in range(self.batch_size):
                x[i] = torch.Tensor(sliced_numpy_song[i*hop:i*hop+self.input_length]).unsqueeze(0)
            audio_tensor=x
            x = Variable(audio_tensor)
            out = self.model(x)
            out = out.detach().cpu()
            mean_out = np.array(out).mean(axis=0)
            song_features=mean_out

            # Create a new row for the current song and its features
            new_row = pd.Series([row['track_uri']] + list(song_features), index=tagged_songs.columns)

            # Append the new row to the tagged_songs DataFrame
            tagged_songs = tagged_songs.append(new_row, ignore_index=True)

        # Save the tagged_songs DataFrame as a CSV file
        tagged_songs.to_csv('tagged_songs.csv', index=False)

def main():
    # Load your dataset, model and set other required parameters here
    numpy_songs_path = DATA_NUMPY_PATH
    model_name = 'crnn'
    fs = 16000
    batch_size = 16
    full_dataset_path = FULL_DATASET_PATH

    music_tagger = MusicTagger(full_dataset_path, numpy_songs_path, model_name, fs, batch_size)
    music_tagger.make_tags_csv()

if __name__ == '__main__':
    main()
