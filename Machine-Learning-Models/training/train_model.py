# coding: utf-8
import os
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
#from torch.utils.tensorboard import SummaryWriter
# Import Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


import model as Model
from model_helpers import get_scores
from paths import BINARY_PATH,VALID_PATH

class CustomFormatter(ticker.ScalarFormatter):
    def __init__(self, useOffset=True, useMathText=True):
        super().__init__(useOffset=useOffset, useMathText=useMathText)

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = 0

    def _set_format(self):
        self.format = "%1.2f"

    def __call__(self, x, pos=None):
        if x.is_integer():
            self.format = "%1.0f"
        else:
            self.format = "%1.2f"
        return super().__call__(x, pos)


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

        # Initialize lists to store values for plotting
        self.train_losses = []
        self.validation_losses = []
        self.validation_roc_aucs = []
        self.validation_pr_aucs = []
        self.relative_loss_reductions = []

        # Define file paths
        self.binary_path=BINARY_PATH
        self.valid_path=VALID_PATH
        # Build model and load data
        self.load_csvs()
        self.build_model()

        # Initialize Tensorboard writer
        #self.writer = SummaryWriter()

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


            #self.writer.add_scalar('Loss/train', batch_loss.item(), epoch)
            self.train_losses.append(batch_loss.item())

            # Compute the relative loss reduction and add it to the list
            if len(self.train_losses) > 1:
                last_loss = self.train_losses[-2]
                current_loss = self.train_losses[-1]
                relative_loss_reduction = (last_loss - current_loss) / last_loss
                self.relative_loss_reductions.append(relative_loss_reduction)

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
        
        # Plot the values after training
        self.plot_values()

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
        #self.writer.add_scalar('Loss/Validation', validation_loss, epoch)
        #self.writer.add_scalar('AUC/ROC', validation_roc_auc, epoch)
        #self.writer.add_scalar('AUC/PR', validation_pr_auc, epoch)
        self.validation_losses.append(validation_loss)
        self.validation_roc_aucs.append(validation_roc_auc)
        self.validation_pr_aucs.append(validation_pr_auc)

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
            

    def plot_values(self):
        """
        Plot the training loss, relative loss reduction, validation loss, ROC AUC, and PR AUC.
        """
        epochs = range(1, self.number_of_epochs + 1)
        loss_reduction_epochs = range(2, self.number_of_epochs + 1)
        # Set a seaborn style for better visuals
        sns.set(style='whitegrid')

        # Training Loss
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(epochs, self.train_losses, color='red', label='Training Loss')
        ax1.set(xlabel='Epoch', ylabel='Loss')
        ax1.set_title('Training Loss')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_formatter(CustomFormatter())

        # Relative Loss Reduction
        fig2, ax2 = plt.subplots(figsize=(9, 6))
        ax2.plot(loss_reduction_epochs, self.relative_loss_reductions, color='purple', label='Relative Loss Reduction')
        ax2.set(xlabel='Epoch', ylabel='Relative Loss Reduction')
        ax2.set_title('Relative Loss Reduction')
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_formatter(CustomFormatter())

        # Validation Loss
        fig3, ax3 = plt.subplots(figsize=(9, 6))
        ax3.plot(epochs, self.validation_losses, color='red', label='Validation Loss')
        ax3.set(xlabel='Epoch', ylabel='Loss')
        ax3.set_title('Validation Loss')
        ax3.legend(loc='upper right')
        ax3.xaxis.set_major_formatter(CustomFormatter())

        # ROC AUC
        fig4, ax4 = plt.subplots(figsize=(9, 6))
        ax4.plot(epochs, self.validation_roc_aucs, color='blue', label='ROC AUC')
        ax4.set(xlabel='Epoch', ylabel='ROC AUC')
        ax4.set_title('ROC AUC')
        ax4.legend(loc='upper right')
        ax4.xaxis.set_major_formatter(CustomFormatter())

        # PR AUC
        fig5, ax5 = plt.subplots(figsize=(9, 6))
        ax5.plot(epochs, self.validation_pr_aucs, color='purple', label='PR AUC')
        ax5.set(xlabel='Epoch', ylabel='PR AUC')
        ax5.set_title('PR AUC')
        ax5.legend(loc='upper right')
        ax5.xaxis.set_major_formatter(CustomFormatter())

        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        fig4.tight_layout()
        fig5.tight_layout()

        # Save the plots as high-quality images
        plt.figure(fig1.number)
        plt.savefig('training_loss.png', dpi=300)
        plt.figure(fig2.number)
        plt.savefig('relative_loss_reduction.png', dpi=300)
        plt.figure(fig3.number)
        plt.savefig('validation_loss.png', dpi=300)
        plt.figure(fig4.number)
        plt.savefig('roc_auc.png', dpi=300)
        plt.figure(fig5.number)
        plt.savefig('pr_auc.png', dpi=300)

        # Display the plots
        plt.show()

