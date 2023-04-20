# coding: utf-8
import numpy as np
import pandas as pd
import pandas as pd
import click
import torch
import matplotlib.pyplot as plt
import os

from ..training import model as Model
from ..training.model_helpers import get_scores
from augmentation_helpers import DataAugmentation
from paths import BINARY_PATH,TEST_PATH,MP3_DATA_PATH,MODEL_LOAD_PATH

class Predict(object):
    def __init__(self, config):
        self.model_path = config.model_path
        self.data_path = config.data_path
        self.batch_size = config.batch_size


        self.binary_path=BINARY_PATH
        self.test_path=TEST_PATH
        self.load_csvs()
        self.build_model()
    
    def load_csvs(self):
        self.binary = np.load(self.binary_path)
        self.test_list  = np.load(self.test_path)

    def build_model(self):
        """
        Build the PyTorch model.
        """
        self.input_length = 29 * 16000
        self.model=Model.CRNN()

        # load model
        S = torch.load(self.model_path, map_location=torch.device('cpu'))
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def plot_scores(self,scores, title, xlabel, ylabel):
        x_values = [score[0] for score in scores]
        y_values = [score[1] for score in scores]
        
        plt.plot(x_values, y_values, marker='o')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
    

    def time_stretch_scores(self, save_path='./'):
        roc_auc_scores = []
        pr_auc_scores = []
        loss_scores = []
        
        for rate in np.arange(0.76, 1.41, 0.1):
            self.time_stretch_augmentation = DataAugmentation('time_stretch', rate)
            loss,roc_auc,pr_auc = get_scores(
                mode='robustness_studies',
                model=self.model,
                list_to_iterate_on=self.test_list,
                batch_size=self.batch_size,
                binary=self.binary,
                data_path=self.data_path,
                input_length=self.input_length,
                augmentation_class=self.time_stretch_augmentation
            )
            roc_auc_scores.append((rate, roc_auc))
            pr_auc_scores.append((rate, pr_auc))
            loss_scores.append((rate, loss))
        
        # Plot ROC AUC scores
        self.plot_scores(
            scores=roc_auc_scores,
            title='ROC AUC Scores',
            xlabel='Stretch Factor',
            ylabel='ROC AUC'
        )

        # Save ROC AUC figure
        if save_path is not None:
            roc_auc_fig_path = os.path.join(save_path, 'roc_auc_scores.png')
            plt.savefig(roc_auc_fig_path)

        # Plot PR AUC scores
        self.plot_scores(
            scores=pr_auc_scores,
            title='PR AUC Scores',
            xlabel='Stretch Factor',
            ylabel='PR AUC'
        )

        # Save PR AUC figure
        if save_path is not None:
            pr_auc_fig_path = os.path.join(save_path, 'pr_auc_scores.png')
            plt.savefig(pr_auc_fig_path)

        # Plot Loss scores
        self.plot_scores(
            scores=loss_scores,
            title='Loss Scores',
            xlabel='Stretch Factor',
            ylabel='Loss'
        )

        # Save Loss figure
        if save_path is not None:
            loss_fig_path = os.path.join(save_path, 'loss_scores.png')
            plt.savefig(loss_fig_path)

        
    def pitch_shift_augmentation_scores(self,save_path='./'):
        roc_auc_scores = []
        pr_auc_scores = []
        loss_scores = []
        
        for rate in np.arange(-2, 2, 0.2):
            self.pitch_shift_augmentation = DataAugmentation('pitch_shift', rate)
            loss,roc_auc,pr_auc =get_scores(mode='robustness_studies',model=self.model,list_to_iterate_on=self.test_list,
                                                                                            batch_size=self.batch_size,binary=self.binary,
                                                                                            data_path=self.data_path,input_length=self.input_length,
                                                                                            augmentation_class=self.pitch_shift_augmentation)
            roc_auc_scores.append((rate, roc_auc))
            pr_auc_scores.append((rate, pr_auc))
            loss_scores.append((rate, loss))
        
        # Plot ROC AUC scores
        self.plot_scores(
            scores=roc_auc_scores,
            title='ROC AUC Scores',
            xlabel='Pitch Shift',
            ylabel='ROC AUC'
        )

        # Save ROC AUC figure
        if save_path is not None:
            roc_auc_fig_path = os.path.join(save_path, 'roc_auc_scores.png')
            plt.savefig(roc_auc_fig_path)

        # Plot PR AUC scores
        self.plot_scores(
            scores=pr_auc_scores,
            title='PR AUC Scores',
            xlabel='Pitch Shift',
            ylabel='PR AUC'
        )

        # Save PR AUC figure
        if save_path is not None:
            pr_auc_fig_path = os.path.join(save_path, 'pr_auc_scores.png')
            plt.savefig(pr_auc_fig_path)

        # Plot Loss scores
        self.plot_scores(
            scores=loss_scores,
            title='Loss Scores',
            xlabel='Pitch Shift',
            ylabel='Loss'
        )

        # Save Loss figure
        if save_path is not None:
            loss_fig_path = os.path.join(save_path, 'loss_scores.png')
            plt.savefig(loss_fig_path)

        

    def test(self):
        self.time_stretch_scores()
        self.pitch_shift_augmentation_scores()
        self.dynamic_range_compression_augmentation_scores()
                                                                                                                                                                 



@click.command()
@click.option('--batch_size', type=int, default=16, help='batch size for the model')
@click.option('--model_path', type=str, default=MODEL_LOAD_PATH, help='path to saved model')
@click.option('--data_path', type=str, default=MP3_DATA_PATH, help='path to data directory')
def run(batch_size, model_path, data_path):
    """
    This script runs the prediction process.

    Args:
        batch_size: batch size for the model
        model_path: path to saved model
        data_path: path to data directory
    """

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    config = Config(
        batch_size=batch_size,
        model_path=model_path,
        data_path=data_path,
    )


    p = Predict(config)
    p.test()

if __name__ == '__main__':
    run()










