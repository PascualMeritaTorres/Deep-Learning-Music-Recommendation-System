# coding: utf-8
import os
import numpy as np
from torch.utils import data
import pandas as pd

from paths import TRAIN_PATH,VALID_PATH,TEST_PATH,BINARY_PATH

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
		self.binary = np.load(BINARY_PATH)
		self.set_dataset()

	
	def __getitem__(self, index):
		"""
		Get an audio file and its corresponding binary tag.

		:param index: Index of the audio file.
		:type index: int
		:return: Tuple of the audio signal and the binary tag.
		:rtype: (np.ndarray, np.ndarray)
		"""
		numpy_song, tags = self.get_numpy_file(index)
		return numpy_song.astype('float32'), tags.astype('float32')
	
	def __len__(self):
		"""
		Get the number of audio files.

		:return: Length of the dataset.
		:rtype: int
		"""
		return len(self.file)

	def set_dataset(self):
		if self.split == 'TRAIN':
			self.file = np.load(TRAIN_PATH)
		elif self.split == 'VALID':
			self.file = np.load(VALID_PATH)
		elif self.split == 'TEST':
			self.file = np.load(TEST_PATH)


	def get_numpy_file(self, index):
		"""
		Get a numpy array of an audio file and its corresponding binary tag.

		:param index: Index of the audio file.
		:type index: int
		:return: Tuple of the audio signal and the binary tag.
		:rtype: (np.ndarray, np.ndarray)
		"""
		index, song_file_name = self.file[index].split('\t')
		numpy_path = os.path.join(self.root,'mtat', 'npy', song_file_name.split('/')[1][:-3]) + 'npy'#Contruct the file path for the corresponding .npy file by joining the root path, and npy, and the file name without the extension
		numpy_song = np.load(numpy_path, mmap_mode='r') #load the npy file into memory
		random_idx = int(np.floor(np.random.random(1) * (len(numpy_song)-self.input_length))) #generate a random integer to select a random starting point in the numpy array
		sliced_numpy_song = np.array(numpy_song[random_idx:random_idx+self.input_length]) #slice the original .npy array, starting at the random index and ending at the random index+input_length
		tags = self.binary[int(index)] #retrieve the binary labels corresponding to the index
		return sliced_numpy_song, tags #return a npy song, tags tuple


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
								  num_workers=parallel_threads, #n. of subprocesses to use for data loading
								  pin_memory=True)
								  
	return data_loader

