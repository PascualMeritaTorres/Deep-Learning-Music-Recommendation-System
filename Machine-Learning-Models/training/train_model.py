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
from torch.utils.tensorboard import SummaryWriter

import model as Model
from model_helpers import get_scores

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
        self.binary_path='./../../Dataset-Creation-And-Preprocessing/our_data/binary.npy'
        self.valid_path='./../../Dataset-Creation-And-Preprocessing/our_data/val.csv'

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
        if len(self.model_load_path) > 1:
            self.load_saved_model(self.model_load_path)

        # Set the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-4)
        
    def train(self):
        """
        Train the PyTorch model.
        """
        # Initialize training variables
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = nn.BCELoss() #Binary Cross Entropy Loss
        best_metric = 0 #stores the best result we have gotten so far
        drop_counter = 0 #used to decide which optimizer to use

        # Training loop
        for epoch in range(self.number_of_epochs):
            ctr = 0 
            drop_counter += 1 #increase the drop counter
            self.model = self.model.train() #Set the model in training mode

            # Iterate over data batches
            for x, y in self.data_loader: #x is the .npy of the song, y is the binary tags
                ctr += 1

                # Forward pass
                x=Variable(x)
                y=Variable(y)
                out = self.model(x) #Pass the numpy array into the model

                # Backward pass
                loss = reconst_loss(out, y) #compare the actual tags (y) with the ones outputted from the model
                self.optimizer.zero_grad() #Set all gradients to 0
                loss.backward() #Compute gradient
                self.optimizer.step() #Update parameters

                # Log the training progress
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # Perform validation and update the best metric
            best_metric = self.get_validation(best_metric, epoch) #Calculate the Binary Cross Entropy loss of our current model using the validation set

            # Update the optimizer accordingly
            current_optimizer, drop_counter = self.modify_optimizer_accordingly(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def modify_optimizer_accordingly(self, current_optimizer, drop_counter):
        """
        Update the optimizer based on the value of the drop counter.

        Parameters:
        - current_optimizer: the name of the current optimizer
        - drop_counter: a counter used to decide which optimizer to use

        Returns:
        - current_optimizer: the name of the current optimizer
        - drop_counter: the updated value of the drop counter
        """
        
        # Update the model with the best saved model
        self.load_saved_model(os.path.join(self.model_save_path, 'best_model.pth'))

        # Update the optimizer based on the drop_counter value
        if current_optimizer == 'adam' and drop_counter == 80:
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001,
                                            momentum=0.9, weight_decay=0.0001,
                                            nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def print_log(self, epoch, ctr, loss, start_t):
        """
        Print the training progress log.

        Parameters:
        - epoch: the current epoch
        - ctr: the current iteration
        - loss: the current loss value
        - start_t: the start time of training
        """
        
        # Print the training progress log
        if (ctr) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

    def get_validation(self, best_metric, epoch):
        """
        Perform validation and update the best metric.

        Parameters:
        - best_metric: the current best validation metric
        - epoch: the current epoch

        Returns:
        - best_metric: the updated best validation metric
        """
        
        # Get validation scores and loss
        score,loss,roc_auc,pr_auc= get_scores(mode='training_validation',model=self.model,list_to_iterate_on=self.valid_list,batch_size=self.batch_size,
                                              binary=self.binary,data_path=self.data_path,input_length=self.input_length)
        self.writer.add_scalar('Loss/valid', loss, epoch)
        self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', pr_auc, epoch)

        # Update the best metric and save the best model if necessary
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                    os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric