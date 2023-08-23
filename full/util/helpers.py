import os
import torch
import numpy as np
import math


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)


def list_of_distances_3d(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=3).unsqueeze(dim=4) - torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0)) ** 2, dim=2)


def list_of_distances_3d_dot(X, Y):
    return 2 - (torch.sum(torch.unsqueeze(X, dim=3).unsqueeze(dim=4) * torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0), dim=2) + 1)    ####### [0, 2]


def list_of_similarities_3d_dot(X, Y):
    return torch.sum(torch.unsqueeze(X, dim=3).unsqueeze(dim=4) * torch.unsqueeze(Y.permute(2, 0, 1), dim=0).unsqueeze(0), dim=2)    #######


def make_one_hot(target, target_one_hot):
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1