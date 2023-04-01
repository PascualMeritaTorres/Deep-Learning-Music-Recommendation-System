# coding: utf-8
import numpy as np
import pandas as pd
import pandas as pd
import click
import torch

import model as Model
from augmentation_helpers import DataAugmentation
from model_helpers import get_scores
from paths import BINARY_PATH,TEST_PATH,DATA_PATH

class Predict(object):
    def __init__(self, config):
        self.aug=DataAugmentation(config.augmentation_type,config.rate)
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size


        self.binary_path=BINARY_PATH
        self.test_path=TEST_PATH
        self.load_csvs()
        self.build_model()
    
    def load_csvs(self):
        self.binary = np.load(self.binary_path)
        # Load the CSV file using Pandas
        data = pd.read_csv(self.test_path)
        # Convert the data to a NumPy array
        self.test_list = data.to_numpy()

    def build_model(self):
        """
        Build the PyTorch model.
        """
        if self.model_name == 'fcn':
            self.model= Model.FCN()
        elif self.model_name == 'crnn':
            self.model=Model.CRNN()
        elif self.model_name == 'short':
            self.model=Model.ShortChunkCNN()
        elif self.model_name == 'short_res':
            self.model=Model.ShortChunkCNN_Res()

        # Load the pretrained model if available
        if len(self.model_load_path) > 1:
            S = torch.load(self.model_load_path)
            self.model.load_state_dict(S)

    def test(self):
        get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                    batch_size=self.batch_size,binary=self.binary,
                    data_path=self.data_path,input_length=self.input_length,
                    augmentation_class=self.aug)



@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='fcn', help='type of model to use')
@click.option('--batch_size', type=int, default=16, help='batch size for the model')
@click.option('--model_path', type=str, default='.', help='path to saved model')
@click.option('--data_path', type=str, default=DATA_PATH, help='path to data directory')
@click.option('--augmentation_type', type=click.Choice(['time_stretch', 'pitch_shift', 'dynamic_range_compression', 'white_noise']), default='time_stretch', help='type of augmentation to apply')
@click.option('--rate', type=float, default=0, help='rate parameter for the augmentation')
def run(model_name, batch_size, model_path, data_path, augmentation_type, rate):
    """
    This script runs the prediction process.

    Args:
        model_name: type of model to use
        batch_size: batch size for the model
        model_path: path to saved model
        data_path: path to data directory
        augmentation_type: type of augmentation to apply
        rate: rate parameter for the augmentation
    """

    config = {
        'model_name': model_name,
        'batch_size': batch_size,
        'model_path': model_path,
        'data_path': data_path,
        'augmentation_type': augmentation_type,
        'rate': rate,
    }

    p = Predict(config)
    p.test()

if __name__ == '__main__':
    run()










