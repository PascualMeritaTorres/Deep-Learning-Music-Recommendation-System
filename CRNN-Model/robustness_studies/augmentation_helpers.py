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
            Type of data augmentation to apply. Available options: "time_stretch", "pitch_shift".
        rate : float
            Rate of augmentation to apply. Its meaning varies depending on the type of augmentation.

        Returns:
        --------
        numpy.ndarray
            Modified audio signal.
        '''
        self.augmentation_type = augmentation_type
        self.rate = rate
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

    def time_stretch(self,x, rate):
        '''
        Modifies audio signal by time-stretching it.

        Parameters:
        -----------
        x : numpy.ndarray
            Input audio signal.
        rate : float
            Stretch factor. Values should be within the range [0.7071, 1.414].

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
            Shift factor. Values should be within the range [-2, 2].

        Returns:
        --------
        numpy.ndarray
            Pitch-shifted audio signal.
        '''
        return librosa.effects.pitch_shift(x, 16000, rate)


 

