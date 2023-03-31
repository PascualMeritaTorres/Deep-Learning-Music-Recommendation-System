# coding: utf-8
#import folium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio

class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2): #shape is the kernel size fro convolution #pooling is the kernel size of the pooling
        #In init we define the layers we are going to use
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        #In forward we define how the layers are stacked together
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2): #the kernel size is the shape
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2) #first convolution layer
        self.bn_1 = nn.BatchNorm2d(output_channels) #first batch normalization layer
        self.conv_2 = nn.Conv2d(output_channels, output_channels, shape, padding=shape//2) #second convolution layer
        self.bn_2 = nn.BatchNorm2d(output_channels) #second batch normalization layer (which is the same as the first one)



        # residual
        '''
        These lines define the residual connection for the residual block.
        If the stride is not 1 or the input and output channels are different, a third convolution layer, self.conv_3,
        and its batch normalization layer, self.bn_3, are defined. These layers are used to match the dimensions 
        of the input tensor and the output tensor of the block, allowing them to be added together.
        '''
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out
    


class FCN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000, #sample rate of 16kHz menans the ANN will be able to see sounds up to 8kHz
                n_fft=512, #size of fourier transform
                f_min=0.0, #minimum frequency
                f_max=8000.0, #maximum frequency (IT MUST BE HALF OF THE SAMPLE RATE)
                n_mels=96, #number of mel filterbanks
                n_class=24): #This is the number of output classes for the model
        super(FCN, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        # Compute the Mel Spectogram from the audio signal
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels) 
        self.to_db = torchaudio.transforms.AmplitudeToDB() #Turns a tensor from the power/amplitude scale to the decibel scale. The decibel scale is more appropriate for human perception of sound intensity, since it is logarithmic and more closely matches the way that our ears perceive sound.
        self.spec_bn = nn.BatchNorm2d(1) #layer that normalizes the output of the spectrogram before passing it to the rest of the network. Batch normalization helps to prevent overfitting and can improve the speed and stability of training.

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2,4)) #1 input channel (single channel Mel-Spectogram) and 64 output channels (64 feature maps, each corresponding to a learned feature)
        self.layer2 = Conv_2d(64, 128, pooling=(2,4)) #64 input channels and 128 output channels
        self.layer3 = Conv_2d(128, 128, pooling=(2,4))
        self.layer4 = Conv_2d(128, 128, pooling=(3,5))
        self.layer5 = Conv_2d(128, 64, pooling=(4,4))

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(64, n_class)

    def forward(self, x):
        # Spectrogram
        x=self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # Dense
        x = x.view(x.size(0), -1) #Flattens the tensor
        x = self.dropout(x) #Apply a dropout of 0.5 to prevent overfitting
        x = self.dense(x) #This is a linear layer (not a feedforward NN) that takes the flattened output of layer5 as input and produces an output with n_class dimensions. 
        x = nn.Sigmoid()(x) #Gets the output of the dense layer and puts it in a scale of 0-1 (before it could be negative and whatsoever)
        return x


class CRNN(nn.Module):
    '''
    Choi et al. 2017
    Convolution recurrent neural networks for music classification.
    Feature extraction with CNN + temporal summary with RNN
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(CRNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, 64, pooling=(2,2))
        self.layer2 = Conv_2d(64, 128, pooling=(3,3))
        self.layer3 = Conv_2d(128, 128, pooling=(4,4))
        self.layer4 = Conv_2d(128, 128, pooling=(4,4))

        # RNN
        self.layer5 = nn.GRU(128, 32, 2, batch_first=True)

        # Dense
        self.dropout = nn.Dropout(0.5)
        self.dense = nn.Linear(32, 50)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # RNN
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        x, _ = self.layer5(x)
        x = x[:, -1, :]

        # Dense
        x = self.dropout(x)
        x = self.dense(x)
        x = nn.Sigmoid()(x)

        return x





class ShortChunkCNN(nn.Module):
    '''
    Short-chunk CNN architecture.
    So-called vgg-ish model with a small receptive field.
    Deeper layers, smaller pooling (2x2).
    '''
    def __init__(self,
                n_channels=128,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=128,
                n_class=50):
        super(ShortChunkCNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Conv_2d(1, n_channels, pooling=2)
        self.layer2 = Conv_2d(n_channels, n_channels, pooling=2)
        self.layer3 = Conv_2d(n_channels, n_channels*2, pooling=2)
        self.layer4 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer5 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer6 = Conv_2d(n_channels*2, n_channels*2, pooling=2)
        self.layer7 = Conv_2d(n_channels*2, n_channels*4, pooling=2)

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
                n_class=50):
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
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels*2, stride=2)
        self.layer4 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer5 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer6 = Res_2d(n_channels*2, n_channels*2, stride=2)
        self.layer7 = Res_2d(n_channels*2, n_channels*4, stride=2)

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


