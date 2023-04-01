# coding: utf-8
import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import model as Model
from model_helpers import get_scores
from paths import BINARY_PATH,VALID_PATH

class TrainingLogic(object):
    def __init__(self, data_loader, config):
        """
        Initialize the Solver object.

        Parameters:
        - data_loader: a PyTorch DataLoader object for loading the training data
        - config: a configuration object containing various hyperparameters for the model and training process
        """
        # Initialize variables from the configuration
        self.data_loader = data_loader
        self.data_path = config.data_path
        self.input_length = config.input_length
        self.number_of_epochs = config.number_of_epochs
        self.learning_rate = config.learning_rate
        self.use_tensorboard = config.use_tensorboard
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_name = config.model_name

        # Define file paths
        self.binary_path=BINARY_PATH
        self.valid_path=VALID_PATH
        # Build model and load data
        self.load_csvs()
        self.build_model()

        # Initialize Tensorboard writer
        self.writer = SummaryWriter()

    def load_csvs(self):
        """
        Load the binary.npy and validation CSV files and convert them to NumPy arrays.
        """
        # Load the binary.npy file
        self.binary = np.load(self.binary_path)
        
        # Load the validation CSV file using Pandas
        data = pd.read_csv(self.valid_path)
        
        # Convert the data to a NumPy array
        self.valid_list = data.to_numpy()

    def load_saved_model(self, filename):
        """
        Load a saved PyTorch model.

        Parameters:
        - filename: the file path of the saved model
        """
        # Load the saved model
        S = torch.load(filename)
        
        # Assign the value of the mel_scale to the fb parameter of the model if available
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        
        # Load the state dictionary into the model
        self.model.load_state_dict(S)

    def build_model(self):
        """
        Build the PyTorch model and set the optimizer.
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
        # if len(self.model_load_path) > 1:
        #     self.load_saved_model(self.model_load_path)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-4)

        # Initialize the learning rate scheduler
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)

        
    def train(self):
        """
        Train the PyTorch model.
        """
        # Initialize training variables
        start_time = time.time()
        reconstruction_loss = nn.BCELoss() # Binary Cross Entropy Loss
        best_metric_so_far = 0 # stores the best result obtained so far
        drop_counter = 0 # used to decide which optimizer to use

        # Training loop
        for epoch in range(self.number_of_epochs):
            batch_counter = 0 
            self.model = self.model.train() # set the model in training mode

            # Iterate over data batches
            for x, y in self.data_loader: # x is the sliced .npy of the song, y is the binary tags
                batch_counter += 1

                # Forward pass
                x_input = Variable(x)
                y_target = Variable(y)
                model_output = self.model(x_input) # Pass the numpy array into the model

                # Backward pass
                batch_loss = reconstruction_loss(model_output, y_target) # compare the actual tags (y) with the ones outputted from the model
                self.optimizer.zero_grad() # Set all gradients to 0
                batch_loss.backward() # Compute gradient
                self.optimizer.step() # Update parameters

                # Log the training progress
                self.print_log(epoch, batch_counter, batch_loss, start_time)


            self.writer.add_scalar('Loss/train', batch_loss.item(), epoch)
            # Perform validation and update the best metric
            best_metric_so_far = self.get_validation(best_metric_so_far, epoch) # Calculate the Binary Cross Entropy loss of our current model using the validation set

            # Switch to SGD after 80 epochs and set the initial learning rate
            if epoch == 79:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001,
                                                 momentum=0.9, weight_decay=0.0001,
                                                 nesterov=True)
                self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.1)

            # Update the learning rate using the scheduler
            self.scheduler.step()

        print("[%s] Train finished. Elapsed: %s"
            % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                datetime.timedelta(seconds=time.time() - start_time)))


    def get_validation(self, best_validation_metric, epoch):
        """
        Perform validation and update the best validation metric.

        Parameters:
        - best_validation_metric: the current best validation metric
        - epoch: the current epoch

        Returns:
        - updated_best_validation_metric: the updated best validation metric
        """
        
        # Get validation scores and loss
        validation_score, validation_loss, validation_roc_auc, validation_pr_auc = get_scores(mode='training_validation',
                                                                                                model=self.model,
                                                                                                list_to_iterate_on=self.valid_list,
                                                                                                batch_size=self.batch_size,
                                                                                                binary=self.binary,
                                                                                                data_path=self.data_path,
                                                                                                input_length=self.input_length)
        self.writer.add_scalar('Loss/Validation', validation_loss, epoch)
        self.writer.add_scalar('AUC/ROC', validation_roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', validation_pr_auc, epoch)

        # Update the best validation metric and save the best model if necessary
        if validation_score > best_validation_metric:
            print('best model!')
            best_validation_metric = validation_score
            torch.save(self.model.state_dict(),
                    os.path.join(self.model_save_path, 'best_model.pth'))
        return best_validation_metric


    def print_log(self, epoch, iteration, loss, start_time):
        """
        Print the training progress log.

        Parameters:
        - epoch: the current epoch
        - iteration: the current iteration
        - loss: the current loss value
        - start_time: the start time of training
        """
        
        # Print the training progress log
        if (iteration) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] Train Loss: %.4f Elapsed Time: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.number_of_epochs, iteration, len(self.data_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_time)))
