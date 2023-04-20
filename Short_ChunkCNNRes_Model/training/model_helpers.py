import os
import numpy as np
import tqdm
from sklearn import metrics
import torch
import torch.nn as nn
from torch.autograd import Variable

from ..robustness_studies.augmentation_helpers import DataAugmentation


def get_auc(est_array, gt_array):
    '''
    Computes the area under the curve of the ROC and precision-recall curves.
    Parameters:
    -----------
    est_array : numpy.ndarray
        Predicted labels.
    gt_array : numpy.ndarray
        Ground truth labels.

    Returns:
    --------
    tuple
        A tuple containing the ROC and precision-recall AUC scores.
    '''
    if len(np.unique(gt_array)) <= 1:
        # Custom handling for a single class
        print("Only one unique class in ground truth labels. Skipping ROC AUC calculation.")
        roc_aucs = -1
    else:
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    print('roc_auc: %.4f' % roc_aucs)
    print('pr_auc: %.4f' % pr_aucs)
    return roc_aucs, pr_aucs

def get_tensor(fn,data_path,input_length,batch_size,mode,augmentation_class):
    '''
    Loads an audio file, applies data augmentation if required, and returns it as a PyTorch tensor.
    Parameters:
    -----------
    fn : list
        List containing the audio file's index and filename.
    data_path : str
        Path to the directory where the audio files are stored.
    input_length : int
        Length of the audio signal.
    batch_size : int
        Batch size.
    mode : str
        Mode of operation. Available options: "evaluation", "training_validation", "robustness_studies".
    augmentation_class : DataAugmentation
        Data augmentation object to use for applying augmentation.

    Returns:
    --------
    torch.Tensor
    Audio signal as a PyTorch tensor.
    '''
    
    # load audio
    npy_path = os.path.join(data_path, 'npy', fn[1] + '.npy')
    raw = np.load(npy_path, mmap_mode='r')

    if mode=='robustness_studies':
        raw = augmentation_class.modify(raw) #USE DATA AUGMENTATION ON THE FILE

    # split chunk
    length = len(raw)
    hop = (length - input_length) // batch_size
    x = torch.zeros(batch_size, input_length)
    for i in range(batch_size):
        x[i] = torch.Tensor(raw[i*hop:i*hop+input_length]).unsqueeze(0)
    return x

def get_scores(mode,model,list_to_iterate_on,batch_size,binary,data_path,input_length, augmentation_class=DataAugmentation('time_stretch',0)):
    '''
    Computes the loss and evaluation scores (ROC and precision-recall AUC) for a given set of audio files. 
    (When it is called from evaluate_model.y and from robustness_studies.py it is the same)
    (When it is called from train_model.py it is a little bit different)
    Parameters:
    -----------
    mode : str
        Mode of operation. Available options: "evaluation", "training_validation", "robustness_studies".
    model : torch.nn.Module
        PyTorch model to evaluate.
    list_to_iterate_on : list
        List of audio files to evaluate.
    batch_size : int
        Batch size.
    binary : numpy.ndarray
        Binary labels for the audio files.
    data_path : str
        Path to the directory where the audio files are stored.
    input_length : int
        Length of the audio signal.
    augmentation_class : DataAugmentation
        Data augmentation object to use for applying augmentation.

    Returns:
    --------
    tuple
        A tuple containing the evaluation scores and the loss score.
    '''
    model = model.eval() #Turn model evaluation mode on
    est_array = []
    gt_array = []
    losses = []
    reconst_loss = nn.BCELoss()
    for line in tqdm.tqdm(list_to_iterate_on):
        ix=line[0]
        fn=line
        # load and split
        x = get_tensor(fn,data_path,input_length,batch_size,mode,augmentation_class)

        # ground truth
        ground_truth = binary[int(ix)]
        #print("Ground_truth: ", ground_truth)

        # forward
        x=Variable(x)
        y = torch.tensor([ground_truth.astype('float32') for i in range(batch_size)])
        out = model(x)
        loss = reconst_loss(out, y)
        losses.append(float(loss.data))
        out = out.detach().cpu()

        # estimate
        estimated = np.array(out).mean(axis=0)
        est_array.append(estimated)
        gt_array.append(ground_truth)

    est_array, gt_array = np.array(est_array), np.array(gt_array)
    loss = np.mean(losses)
    print('loss: %.4f' % loss)

    if len(np.unique(gt_array)) <= 1:
        # Custom handling for a single class
        print("Only one unique class in ground truth labels. Skipping ROC AUC calculation.")
        roc_auc = -1
        pr_auc = metrics.average_precision_score(gt_array, est_array, average='macro')
    else:
        roc_auc, pr_auc = get_auc(est_array, gt_array)
    
    if mode=='training_validation':
        score = 1 - loss
        return score,loss,roc_auc,pr_auc
    else:
        return loss,roc_auc,pr_auc
