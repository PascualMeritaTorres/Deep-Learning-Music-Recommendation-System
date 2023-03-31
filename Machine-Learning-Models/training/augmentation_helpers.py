import os
import numpy as np
import subprocess
import json
from pkg_resources import resource_filename
import soundfile as psf
import librosa
import tempfile


'''
------------------------------DATA AUGMENTATION METHODS----------------------------------------------------------
'''

'''
Some parts of the deformation are borrowed from SOTA https://github.com/minzwon/sota-music-tagging-models
which in turn borrows the code from MUDA
McFee et al., A software framework for musical data augmentation, 2015
https://github.com/bmcfee/muda
'''
class DataAugmentation(object):
    def __init__(self, augmentation_type,rate):
        '''
        Data augmentation class to modify audio input for training or testing purposes.
        Parameters:
        -----------
        augmentation_type : str
            Type of data augmentation to apply. Available options: "time_stretch", "pitch_shift", "dynamic_range", "white_noise".
        rate : float
            Rate of augmentation to apply. Its meaning varies depending on the type of augmentation.

        Returns:
        --------
        numpy.ndarray
            Modified audio signal.
        '''
        self.augmentation_type = augmentation_type
        self.rate = rate
        self.PRESETS =json.load(open(resource_filename(__name__, "dynamic_compression_presets.json")))
        self.preset_dict = {1: "radio",
                        2: "film standard",
                        3: "film light",
                        4: "music standard",
                        5: "music light",
                        6: "speech"}

    def modify(self,x):
        '''
        Modifies the input audio signal according to the selected data augmentation type.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.

        Returns:
        --------
        numpy.ndarray
            Modified audio signal.
        '''
        if self.augmentation_type == 'time_stretch':
            return self.time_stretch(x, self.rate)
        elif self.augmentation_type == 'pitch_shift':
            return self.pitch_shift(x, self.rate)
        elif self.augmentation_type == 'dynamic_range':
            return self.dynamic_range_compression(x, self.rate)
        elif self.augmentation_type == 'white_noise':
            return self.white_noise(x, self.rate)

    def time_stretch(self,x, rate):
        '''
        Modifies audio signal by time-stretching it.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        rate : float
            Stretch factor. Values should be within the range [2 ** (-.5), 2 ** (.5)].

        Returns:
        --------
        numpy.ndarray
            Time-stretched audio signal.
        '''
        return librosa.effects.time_stretch(x, rate)

    def pitch_shift(self,x, rate):
        '''
        Modifies audio signal by pitch-shifting it.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        rate : float
            Shift factor. Values should be within the range [-1, 1].

        Returns:
        --------
        numpy.ndarray
            Pitch-shifted audio signal.
        '''
        return librosa.effects.pitch_shift(x, 16000, rate)

    def dynamic_range_compression(self,x, rate):
        '''
        Modifies audio signal by compressing its dynamic range.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        rate : float
            Compression factor. Values should be within the range [4, 6].

        Returns:
        --------
        numpy.ndarray
            Compressed audio signal.
        '''
        return self.sox(x, 16000, "compand", *self.PRESETS[self.preset_dict[rate]])

    @staticmethod
    def sox(self,x, fs, *args):
        '''
        Applies the SoX tool to modify the input audio signal.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        fs : int
            Sample rate of the input audio signal.
        *args : list of str
            SoX arguments to apply.

        Returns:
        --------
        numpy.ndarray
            Modified audio signal.
        '''
        assert fs > 0

        fdesc, infile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)
        fdesc, outfile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)

        psf.write(infile, x, fs)

        try:
            arguments = ["sox", infile, outfile, "-q"]
            arguments.extend(args)

            subprocess.check_call(arguments)

            x_out, fs = psf.read(outfile)
            x_out = x_out.T
            if x.ndim == 1:
                x_out = librosa.to_mono(x_out)

        finally:
            os.unlink(infile)
            os.unlink(outfile)

        return x_out

    def white_noise(self,x, rate):
        '''
        Adds white noise to the input audio signal.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        rate : float
            Noise factor. Values should be within the range [0.1, 0.4].

        Returns:
        --------
        numpy.ndarray
            Audio signal with added white noise.
        '''
        n_frames = len(x)
        noise_white = np.random.RandomState().randn(n_frames)
        noise_fft = np.fft.rfft(noise_white)
        values = np.linspace(1, n_frames * 0.5 + 1, n_frames // 2 + 1)
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0
        noise_filtered = noise_fft * colored_filter
        noise = librosa.util.normalize(np.fft.irfft(noise_filtered)) * (x.max())
        if len(noise) < len(x):
            x = x[:len(noise)]
        return (1 - rate) * x + (noise * rate)

