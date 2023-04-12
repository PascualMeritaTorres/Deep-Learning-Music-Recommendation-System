#import folium
import os
import argparse
from train_model import TrainingLogic
from tags_dataset_loader import data_loader
import click

from paths import DATA_PATH,MODEL_LOAD_PATH,MODEL_SAVE_PATH

def configure_and_train(config):
    # Create the path for the models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    # Configure the audio length
    if config.model_name == 'fcn' or config.model_name == 'crnn':
        config.input_length = 29 * 16000
    elif config.model_name in ['short', 'short_res']:
        config.input_length = 59049

    # COnfigure the data loader
    data = data_loader(config.data_path,
                                    config.batch_size,
									split='TRAIN',
                                    parallel_threads=config.parallel_threads,
                                    input_length=config.input_length)
                                    
    logic = TrainingLogic(data, config)
    logic.train()



@click.command()
@click.option('--parallel_threads', type=int, default=8, help='number of parallel threads of the cpu that will be used to train')
@click.option('--model_name', type=click.Choice(['fcn', 'crnn', 'short', 'short_res']), default='crnn', help='name of the model to use')
@click.option('--number_of_epochs', type=int, default=20, help='number of epochs to train')
@click.option('--batch_size', type=int, default=100, help='number of samples passed through to the network at one time')
@click.option('--learning_rate', type=float, default=1e-4, help='learning rate')
@click.option('--use_tensorboard', type=int, default=0, help='use tensorboard for visualization')
@click.option('--model_save_path', type=str, default=MODEL_SAVE_PATH, help='path to save the trained model')
@click.option('--model_load_path', type=str, default=MODEL_LOAD_PATH, help='path to load the saved model')
@click.option('--data_path', type=str, default=DATA_PATH, help='path to the data directory')
@click.option('--log_step', type=int, default=120, help='number of steps to log')
def run(parallel_threads, model_name, number_of_epochs, batch_size, learning_rate, use_tensorboard, model_save_path, model_load_path, data_path, log_step):
    """
    This script trains the model.

    Args:
        parallel_threads: number of parallel threads of the cpu that will be used to train
        model_name: name of the model to use
        number_of_epochs: number of epochs to train
        batch_size: number of samples passed through to the network at one time
        learning_rate: learning rate
        use_tensorboard: use tensorboard for visualization
        model_save_path: path to save the trained models
        model_load_path: path to load the saved model
        data_path: path to the data directory
        log_step: number of steps to log
    """
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
    config = Config(
        parallel_threads=parallel_threads,
        model_name=model_name,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_tensorboard=use_tensorboard,
        model_save_path=model_save_path,
        model_load_path=model_load_path,
        data_path=data_path,
        log_step=log_step
    )
    print(config)
    configure_and_train(config)

if __name__ == '__main__':
    run()

