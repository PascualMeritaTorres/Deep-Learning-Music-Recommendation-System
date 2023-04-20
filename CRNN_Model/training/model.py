# coding: utf-8
#import folium
import torch.nn as nn
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


class CRNN(nn.Module):
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
        self.conv_block1 = Conv2dBlock(1, 64, pooling_size=(2,2))
        self.conv_block2 = Conv2dBlock(64, 128, pooling_size=(3,3))
        self.conv_block3 = Conv2dBlock(128, 128, pooling_size=(4,4))
        self.conv_block4 = Conv2dBlock(128, 128, pooling_size=(4,4))

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