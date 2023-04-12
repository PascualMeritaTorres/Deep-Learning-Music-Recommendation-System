# coding: utf-8
#import folium
import os
import numpy as np
import pandas as pd
import pandas as pd
import click
import torch
import matplotlib.pyplot as plt


import model as Model
from model_helpers import get_scores
from paths import BINARY_PATH,TEST_PATH,DATA_PATH,MODEL_LOAD_PATH

class Evaluate(object):
    def __init__(self, config):
        """
        Initialize the Evaluate object.

        Parameters:
        - config: a configuration object containing various hyperparameters for the model and evaluation process
        """
        self.model_name = config.model_name
        self.model_load_path = config.model_load_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.binary_path=BINARY_PATH
        self.test_path=TEST_PATH
        self.load_csvs()
        self.build_model()

    def load_csvs(self):
        """
        Load the binary.npy and validation CSV files and convert them to NumPy arrays.
        """
        # Load the binary.npy file
        self.binary = np.load(self.binary_path)
        self.test_list = np.load(TEST_PATH)

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
        S = torch.load(self.model_load_path, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def plot_metrics(self,roc_auc, pr_auc, loss):
        metrics = ['ROC AUC', 'PR AUC', 'Loss']
        values = [roc_auc, pr_auc, loss]
        
        fig, ax = plt.subplots()
        ax.bar(metrics, values)
        ax.set_ylabel('Scores')
        ax.set_title('Model Evaluation Metrics')
        
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=8)
        
        plt.show()


    def evaluate(self):
        """
        Perform evaluation

        Returns:
        --------
        tuple
            A tuple containing the evaluation scores and the loss score.
        """
        validation_loss, validation_roc_auc, validation_pr_auc =get_scores(mode='evaluation',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length)

        self.plot_metrics(validation_roc_auc, validation_pr_auc, validation_loss)
        print(validation_loss, validation_roc_auc,validation_pr_auc)


@click.command()
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='crnn', help='name of the model to use')
@click.option('--batch_size', type=int, default=8, help='number of samples passed through to the network at one time')
@click.option('--model_load_path', type=str,default=MODEL_LOAD_PATH, help='path to load the saved model')
@click.option('--data_path', type=str, default=DATA_PATH, help='path to the data directory')
def run(model_name, batch_size,model_load_path, data_path):
    """
    This script trains the model.

    Args:
        model_name: name of the model to use
        batch_size: number of samples passed through to the network at one time
        model_load_path: path to load the saved model
        data_path: path to the data directory
    """
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        model_name=model_name,
        batch_size=batch_size,
        model_load_path=model_load_path,
        data_path=data_path,
    )

    evaluate = Evaluate(config)
    evaluate.evaluate()

if __name__ == '__main__':
    run()







