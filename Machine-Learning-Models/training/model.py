# coding: utf-8
#import folium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio

class Conv2dBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, pooling_size=2):
        """
        A block of convolutional neural network layers with ReLU activation and max pooling.

        Parameters:
        - input_channels: the number of input channels
        - output_channels: the number of output channels
        - kernel_size: the kernel size for convolution
        - stride: the stride for convolution
        - pooling_size: the kernel size for max pooling
        """
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(pooling_size)

    def forward(self, x):
        """
        The forward pass of the block.

        Parameters:
        - x: the input tensor

        Returns:
        - out: the output tensor
        """
        out = self.max_pool(self.relu(self.batch_norm(self.conv(x))))
        return out



class Residual2dBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=2):
        """
        A residual block of convolutional neural network layers with ReLU activation.

        Parameters:
        - input_channels: the number of input channels
        - output_channels: the number of output channels
        - kernel_size: the kernel size for convolution
        - stride: the stride for convolution
        """
        super(Residual2dBlock, self).__init__()

        # Convolution layers
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, padding=kernel_size // 2)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)

        # Residual connection
        if (stride != 1) or (input_channels != output_channels):
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=kernel_size // 2)
            self.batch_norm3 = nn.BatchNorm2d(output_channels)
            self.has_residual_connection = True
        else:
            self.has_residual_connection = False

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        The forward pass of the block.

        Parameters:
        - x: the input tensor

        Returns:
        - out: the output tensor
        """
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.batch_norm2(self.conv2(out))

        if self.has_residual_connection:
            residual = self.batch_norm3(self.conv3(x))
            out = out + residual
        else:
            out = out + x

        out = self.relu(out)
        return out

    


class FCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000, # The sample rate of the input audio signal
                n_fft=512, # The number of points in the Fast Fourier Transform (FFT)
                f_min=0.0, # The lowest frequency in the Mel scale
                f_max=8000.0, # The highest frequency in the Mel scale
                n_mels=96, # The number of Mel frequency bands
                n_classes=32): # The number of output classes for the model
        super(FCN, self).__init__()

        # Mel spectrogram
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                             n_fft=n_fft,
                                                             f_min=f_min,
                                                             f_max=f_max,
                                                             n_mels=n_mels) # Compute the Mel spectrogram from the audio signal
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB() # Converts the amplitude scale to the decibel scale
        self.spec_bn = nn.BatchNorm2d(1) # Batch normalization layer to normalize the Mel spectrogram before passing it to the rest of the network

        # FCN
        self.conv_block1 = Conv2dBlock(1, 64, pooling_size=(2, 4)) # First convolutional block
        self.conv_block2 = Conv2dBlock(64, 128, pooling_size=(2, 4)) # Second convolutional block
        self.conv_block3 = Conv2dBlock(128, 128, pooling_size=(2, 4)) # Third convolutional block
        self.conv_block4 = Conv2dBlock(128, 128, pooling_size=(3, 5)) # Fourth convolutional block
        self.conv_block5 = Conv2dBlock(128, 64, pooling_size=(4, 4)) # Fifth convolutional block

        # Dense
        self.dropout = nn.Dropout(0.5) # Dropout layer to prevent overfitting
        self.fc = nn.Linear(64, n_classes) # Linear layer to produce output with n_classes dimensions

    def forward(self, x):
        # Mel spectrogram
        x = self.mel_spec(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1) # Adds a dimension for the number of channels
        x = self.spec_bn(x)

        # FCN
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        # Dense
        x = x.view(x.size(0), -1) # Flattens the tensor
        x = self.dropout(x) # Applies a dropout of 0.5 to prevent overfitting
        x = self.fc(x) # Computes the output of the linear layer
        x = nn.Sigmoid()(x) # Scales the output to the range [0, 1]
        return x



class CRNN(nn.Module):
   '''
    This is a convolutional recurrent neural network for music classification. The model uses a CNN for feature extraction
    and a RNN for temporal summarization. It was proposed by Choi et al. in 2017.
    Parameters:
    - sample_rate: the sampling rate of the audio files
    - n_fft: the size of the FFT used to compute the spectrogram
    - f_min: the minimum frequency of the Mel filterbanks
    - f_max: the maximum frequency of the Mel filterbanks
    - n_mels: the number of Mel filterbanks to use
    - n_class: the number of output classes for the model
    '''
def __init__(self,
            sample_rate=16000,
            n_fft=512,
            f_min=0.0,
            f_max=8000.0,
            n_mels=96,
            n_class=50):
    super(CRNN, self).__init__()

    # Spectrogram - computes Mel spectrogram for the input audio
    self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                 n_fft=n_fft,
                                                                 f_min=f_min,
                                                                 f_max=f_max,
                                                                 n_mels=n_mels)
    self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    self.spectrogram_bn = nn.BatchNorm2d(1)

    # CNN - feature extraction using four convolutional blocks
    self.conv_block1 = Conv2dBlock(1, 64, pooling=(2,2))
    self.conv_block2 = Conv2dBlock(64, 128, pooling=(3,3))
    self.conv_block3 = Conv2dBlock(128, 128, pooling=(4,4))
    self.conv_block4 = Conv2dBlock(128, 128, pooling=(4,4))

    # RNN - temporal summarization using a Gated Recurrent Unit (GRU)
    self.gru = nn.GRU(128, 32, 2, batch_first=True)

    # Dense - final classification layer with dropout for regularization
    self.dropout = nn.Dropout(0.5)
    self.dense = nn.Linear(32, n_class)

def forward(self, x):
    # Spectrogram - compute and normalize the Mel spectrogram
    x = self.mel_spectrogram(x)
    x = self.amplitude_to_db(x)
    x = x.unsqueeze(1)
    x = self.spectrogram_bn(x)

    # CNN - pass through the four convolutional blocks
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)

    # RNN - prepare input for the GRU and process it
    x = x.squeeze(2)
    x = x.permute(0, 2, 1)
    x, _ = self.gru(x)
    x = x[:, -1, :]

    # Dense - apply dropout and classify using the dense layer
    x = self.dropout(x)
    x = self.dense(x)
    x = nn.Sigmoid()(x)

    return x




class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture for audio classification.
    This is a VGG-like model with a small receptive field.
    It uses deeper layers and smaller pooling (2x2).
    '''
    def __init__(self,
                num_channels=128,
                sample_rate=16000,
                fft_size=512,
                freq_min=0.0,
                freq_max=8000.0,
                num_mel_bins=128,
                num_classes=50):
        super(ShortChunkCNN, self).__init__()

        # MelSpectrogram - computes Mel spectrogram for the input audio
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                     n_fft=fft_size,
                                                                     f_min=freq_min,
                                                                     f_max=freq_max,
                                                                     n_mels=num_mel_bins)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.spectrogram_bn = nn.BatchNorm2d(1)

        # CNN - feature extraction using four convolutional blocks
        self.conv_layer1 = Conv2dBlock(1, num_channels, pooling=2)
        self.conv_layer2 = Conv2dBlock(num_channels, num_channels, pooling=2)
        self.conv_layer3 = Conv2dBlock(num_channels, num_channels*2, pooling=2)
        self.conv_layer4 = Conv2dBlock(num_channels*2, num_channels*2, pooling=2)
        self.conv_layer5 = Conv2dBlock(num_channels*2, num_channels*2, pooling=2)
        self.conv_layer6 = Conv2dBlock(num_channels*2, num_channels*2, pooling=2)
        self.conv_layer7 = Conv2dBlock(num_channels*2, num_channels*4, pooling=2)

        # Dense layers
        self.dense_layer1 = nn.Linear(num_channels*4, num_channels*4)
        self.batch_norm = nn.BatchNorm1d(num_channels*4)
        self.dense_layer2 = nn.Linear(num_channels*4, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)
        x = x.unsqueeze(1)
        x = self.spectrogram_bn(x)

        # CNN layers
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense layers
        x = self.dense_layer1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense_layer2(x)
        x = nn.Sigmoid()(x)

        return x



class ShortChunkCNN_Res(nn.Module):
    '''
    Short-chunk CNN architecture with residual connections.
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=32):
        super(ShortChunkCNN_Res, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Residual2dBlock(1, n_channels, stride=2)
        self.layer2 = Residual2dBlock(n_channels, n_channels, stride=2)
        self.layer3 = Residual2dBlock(n_channels, n_channels*2, stride=2)
        self.layer4 = Residual2dBlock(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Residual2dBlock(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Residual2dBlock(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Residual2dBlock(n_channels*2, n_channels*4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


