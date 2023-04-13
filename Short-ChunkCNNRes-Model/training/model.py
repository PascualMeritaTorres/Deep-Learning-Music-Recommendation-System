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



class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out


class CRNNSOTA(nn.Module):
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
        super(CRNNSOTA, self).__init__()

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