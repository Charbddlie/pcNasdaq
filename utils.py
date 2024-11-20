import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from copy import deepcopy
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from math import pi
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def tanh(xs):
    return torch.tanh(xs)

def linear(x):
    return x

def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0

def linear_deriv(x):
    return torch.ones((1,)).float().to(DEVICE)

def relu(xs):
  return torch.clamp(xs,min=0)

def relu_deriv(xs):
  rel = relu(xs)
  rel[rel>0] = 1
  return rel

def softmax(xs):
  return F.softmax(xs, dim=1) # B, L

def sigmoid(xs):
  return torch.sigmoid(xs)

def sigmoid_deriv(xs):
  return torch.sigmoid(xs) * (torch.ones_like(xs) - torch.sigmoid(xs))

def arctan_deriv(xs):
    alpha = 10.0
    return 1 / (1 + alpha * xs * xs)

### loss functions
def mse_loss(out, label):
      return torch.sum((out-label)**2)

def mse_deriv(out,label):
      return 2 * (out - label)

def cross_entropy_loss(out,label):
      return nn.CrossEntropyLoss(out,label)

def cross_entropy_deriv(out,label):
      return out - label

### Initialization Functions ###
def gaussian_init(W,mean=0.0, std=0.05):
  return W.normal_(mean=0.0,std=0.05)

def zeros_init(W):
  return torch.zeros_like(W)

def kaiming_init(W, a=math.sqrt(5),*kwargs):
  return init.kaiming_uniform_(W, a)

def xavier_init(W):
  return init.xavier_normal_(W)


def generate_ones_and_minus_ones_matrix(rows, cols):
    random_matrix = torch.randint(0, 2, (rows, cols))
    binary_matrix = torch.where(random_matrix == 0, -1 * torch.ones_like(random_matrix), torch.ones_like(random_matrix))
    return binary_matrix.float()

def set_tensor(xs):
    return xs.float().to(DEVICE)

def cosine_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    # Calculate the cosine annealing factor
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    # Compute the current learning rate based on cosine annealing
    lr = min_lr + (initial_lr - min_lr) * cosine_decay
    return lr

def linear_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    epoch = min(epoch, total_epochs)
    # Calculate the linear decay factor
    linear_decay = 1 - epoch / total_epochs
    # Compute the current learning rate based on linear decay
    lr = min_lr + (initial_lr - min_lr) * linear_decay
    return lr

def no_scheduler(epoch, total_epochs, initial_lr, min_lr=0):
    return initial_lr

def rnn_accuracy(model, target_batch):
    accuracy = 0
    L, _, B = target_batch.shape
    for i in range(len(model.y_preds)): # this loop is over the seq_len
        for b in range(B):
            if torch.argmax(target_batch[i,:,b]) ==torch.argmax(model.y_preds[i][:,b]):
                accuracy+=1
    return accuracy / (L * B)


def onehot(arr, vocab_size):
    L, B = arr.shape
    ret = np.zeros([L,vocab_size,B])
    for l in range(L):
        for b in range(B):
            ret[l,int(arr[l,b]),b] = 1
    return ret

def inverse_list_onehot(arr):
    L = len(arr)
    V,B = arr[0].shape
    ret = np.zeros([L,B])
    for l in range(L):
        for b in range(B):
            for v in range(V):
                if arr[l][v,b] == 1:
                    ret[l,b] = v
    return ret

def decode_ypreds(ypreds):
    L = len(ypreds)
    V,B = ypreds[0].shape
    ret = np.zeros([L,B])
    for l in range(L):
        for b in range(B):
            v = torch.argmax(ypreds[l][:,b])
            ret[l,b] =v
    return ret


def inverse_onehot(arr):
    if type(arr) == list:
        return inverse_list_onehot(arr)
    else:
        L,V,B = arr.shape
        ret = np.zeros([L,B])
        for l in range(L):
            for b in range(B):
                for v in range(V):
                    if arr[l,v,b] == 1:
                        ret[l,b] = v
        return ret
    
def RSE(pred, true):
    return torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))


def MSE(pred, true):
    return torch.mean((pred - true) ** 2)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true))


def MSPE(pred, true):
    return torch.mean(torch.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe