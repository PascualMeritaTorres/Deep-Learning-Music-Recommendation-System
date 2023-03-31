# coding: utf-8
import os
import numpy as np
from torch.utils import data
import pandas as pd

class AudioFolder(data.Dataset):
	def __init__(self, root, split, input_length=None):
		"""
		AudioFolder class constructor.
		:param root: Root directory of the audio data.
		:type root: str
		:param split: Data split, can be 'TRAIN', 'VALID' or 'TEST'.
		:type split: str
		:param input_length: Length of audio signal to be loaded.
		:type input_length: int, optional
		"""


		self.root = root
		self.split = split
		self.input_length = input_length
		self.train='./../../Dataset-Creation-And-Preprocessing/our_data/train.csv'
		self.valid='./../../Dataset-Creation-And-Preprocessing/our_data/valid.csv'
		self.test='./../../Dataset-Creation-And-Preprocessing/our_data/test.csv'
		self.binary = np.load('./../../Dataset-Creation-And-Preprocessing/our_data/binary.npy')
		self.set_csv()

	
	def __getitem__(self, index):
		"""
		Get an audio file and its corresponding binary tag.

		:param index: Index of the audio file.
		:type index: int
		:return: Tuple of the audio signal and the binary tag.
		:rtype: (np.ndarray, np.ndarray)
		"""
		numpy_song, features = self.get_numpy_file(index)
		return numpy_song.astype('float32'), features.astype('float32')
	
	def __len__(self):
		"""
		Get the number of audio files.

		:return: Length of the dataset.
		:rtype: int
		"""
		return len(self.csv)

	def set_csv(self):
		if self.split == 'TRAIN':
			# Load the CSV file using Pandas
			data = pd.read_csv(self.train)
			# Convert the data to a NumPy array
			self.csv = data.to_numpy()
		elif self.split == 'VALID':
			# Load the CSV file using Pandas
			data = pd.read_csv(self.valid)
			# Convert the data to a NumPy array
			self.csv = data.to_numpy()
		elif self.split == 'TEST':
			# Load the CSV file using Pandas
			data = pd.read_csv(self.test)
			# Convert the data to a NumPy array
			self.csv = data.to_numpy()
	def get_numpy_file(self, index):
		"""
		Get a numpy array of an audio file and its corresponding binary tag.

		:param index: Index of the audio file.
		:type index: int
		:return: Tuple of the audio signal and the binary tag.
		:rtype: (np.ndarray, np.ndarray)
		"""
		csv_index=self.csv[index][0]
		song_file_name=self.csv[index][1]
		numpy_path = os.path.join(self.root, 'npy', song_file_name + '.npy')#Contruct the file path for the corresponding .npy file by joining the root path, and npy, and the file name without the extension
		numpy_song = np.load(numpy_path, mmap_mode='r') #load the npy file into memory
		random_idx = int(np.floor(np.random.random(1) * (len(numpy_song)-self.input_length))) #generate a random integer to select a random starting point in the numpy array
		sliced_numpy_song = np.array(numpy_song[random_idx:random_idx+self.input_length]) #slice the original .npy array, starting at the random index and ending at the random index+input_length
		features = self.binary[int(csv_index)] #retrieve the binary labels corresponding to the index
		return sliced_numpy_song, features #return a npy song, tags tuple


#This returns a PyTorch DataLoader object. 
def data_loader(root, batch_size, split='TRAIN', parallel_threads=0, input_length=None): 
	"""
	Get a PyTorch DataLoader for the specified data split.
	:param root: Root directory of the audio data.
	:type root: str
	:param batch_size: Number of data samples to load at a time.
	:type batch_size: int
	:param split: Data split, can be 'TRAIN', 'VALID' or 'TEST'.
	:type split: str
	:param parallel_threads: Number of subprocesses to use for data loading.
	:type parallel_threads: int
	:param input_length: Length of audio signal to be loaded.
	:type input_length: int, optional
	:return: PyTorch DataLoader for the specified data split.
	:rtype: data.DataLoader
	"""
	
	data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length), #This is the directory from where the data is loaded
								  batch_size=batch_size, #n. of data samples to load at a time
								  shuffle=True, #shuffle the data at each epoch
								  drop_last=False, #do not discard incomplete batches at the end of each epoch
								  num_workers=parallel_threads) #n. of subprocesses to use for data loading
								  
	return data_loader

