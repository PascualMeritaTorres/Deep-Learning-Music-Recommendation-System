# coding: utf-8
import numpy as np
import pandas as pd
import pandas as pd
import click
import torch

import model as Model
from augmentation_helpers import DataAugmentation
from model_helpers import get_scores
from paths import BINARY_PATH,TEST_PATH,DATA_PATH,MODEL_LOAD_PATH

class Predict(object):
    def __init__(self, config):
        self.time_stretch_augmentation=DataAugmentation('time_stretch',1)
        self.pitch_shift_augmentation=DataAugmentation('pitch_shift',1)
        self.dynamic_range_compression_augmentation=DataAugmentation('dynamic_range',4)
        self.white_noise_augmentation=DataAugmentation('white_noise',0.2)
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
            self.input_length = 29 * 16000
            self.model=Model.FCN()
        elif self.model_name == 'crnn':
            self.input_length = 29 * 16000
            self.model=Model.CRNN()
        elif self.model_name == 'short':
            self.input_length = 59049
            self.model=Model.ShortChunkCNN()
        elif self.model_name == 'short_res':
            self.input_length = 59049
            self.model= Model.ShortChunkCNN_Res()

        # load model
        S = torch.load(self.model_path, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def test(self):
        time_stretch_scores=get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length,
                                                                                            augmentation_class=self.time_stretch_augmentation)

        pitch_shift_augmentation_scores=get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length,
                                                                                            augmentation_class=self.pitch_shift_augmentation)

        dynamic_range_compression_augmentation_scores=get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length,
                                                                                            augmentation_class=self.dynamic_range_compression_augmentation)   

        white_noise_augmentation_scores=get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length,
                                                                                            augmentation_class=self.white_noise_augmentation)             
                                                                                                               
        print(time_stretch_scores,pitch_shift_augmentation_scores,dynamic_range_compression_augmentation_scores,white_noise_augmentation_scores)                                                                                                                                                                   



@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='fcn', help='type of model to use')
@click.option('--batch_size', type=int, default=16, help='batch size for the model')
@click.option('--model_path', type=str, default=MODEL_LOAD_PATH, help='path to saved model')
@click.option('--data_path', type=str, default=DATA_PATH, help='path to data directory')
def run(model_name, batch_size, model_path, data_path):
    """
    This script runs the prediction process.

    Args:
        model_name: type of model to use
        batch_size: batch size for the model
        model_path: path to saved model
        data_path: path to data directory
    """

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name=model_name,
        batch_size=batch_size,
        model_path=model_path,
        data_path=data_path,
    )


    p = Predict(config)
    p.test()

if __name__ == '__main__':
    run()










