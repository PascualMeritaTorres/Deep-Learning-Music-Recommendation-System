import os
import numpy as np
import glob
import librosa
import fire
import tqdm
from pydub import AudioSegment


class Processor:
	def __init__(self):
		self.fs = 16000

	def get_paths(self, data_path):
		##self.files = glob.glob(os.path.join(data_path,'*/*.mp3'))
		self.files = glob.glob(os.path.join(data_path, '**/*.mp3'), recursive=True)
		##self.files = glob.glob(os.path.join(data_path, '**.mp3'), recursive=True)
		#print(self.files)
		self.npy_path = os.path.join(data_path,'npy')
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)

	def get_npy(self, fn):
		x, sr = librosa.load(fn, sr=self.fs)
		#audio = AudioSegment.from_file(fn)
		#x = audio.get_array_of_samples()	
		return x


	def iterate(self, data_path):
		self.get_paths(data_path)
		for fn in tqdm.tqdm(self.files):
			npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy') #creates a file name for a corresponding .npy file to be saved. The fn.split('/')[-1][:-3] extracts the file name and removes the .mp3 extension
			if not os.path.exists(npy_fn): #check if the file already exists
				try:
					x = self.get_npy(fn) #generate the .npy array from the audio file (fn)
					np.save(open(npy_fn, 'wb'), x)
				except (RuntimeError,EOFError) as error:
					# some audio files are broken
					print(fn)
					continue

if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
	#path name = /Users/pascualmeritatorres/Developer/Dissertation/actual-dissertation/PracticalDataScience-ENCA/notebooks/mp3
	#usage= python3 -u mtat_read.py run /Users/pascualmeritatorres/Developer/Dissertation/actual-dissertation/PracticalDataScience-ENCA/notebooks/mp3


