import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import train_util, train_util2
import chainer
import chainer.functions as F
from chainer import Variable
'''
Loss function
author:Fengting Yang 
Mar.1st 2019

We only use "compute_semantic_pos_loss" func. in our final version, best result achieved with weight = 3e-3
'''
def compute_fp(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    prob = prob_in.clone()
    pooled_labxy = train_util.poolfeat(labxy_feat, prob, kernel_size, kernel_size)
    reconstr_feat = train_util.upfeat(pooled_labxy, prob, kernel_size, kernel_size)
    return reconstr_feat

def compute_semantic_pos_loss2(prob_in, labxy_feat,  pos_weight = 0.003,  kernel_size=16):
    # this wrt the slic paper who used sqrt of (mse)

    # rgbxy1_feat: B*50+2*H*W
    # output : B*9*H*w
    # NOTE: this loss is only designed for one level structure

    # todo: currently we assume the downsize scale in x,y direction are always same
    S = kernel_size
    m = pos_weight
    # prob = prob_in.copy()
    labxy_feat = chainer.cuda.to_gpu(labxy_feat)
    b, c, h, w = labxy_feat.shape
    pooled_labxy = train_util2.poolfeat(labxy_feat, prob_in, kernel_size, kernel_size)
    reconstr_feat = train_util2.upfeat(pooled_labxy, prob_in, kernel_size, kernel_size)

    loss_map = reconstr_feat[:,-2:,:,:] - labxy_feat[:,-2:,:,:]

    # self def cross entropy  -- the official one combined softmax
    logit = F.log(reconstr_feat[:, :-2, :, :] + 1e-8)
    loss_sem = - F.sum(logit * labxy_feat[:, :-2, :, :], axis=1) / b
    loss_pos = F.sum(loss_map, axis=1) / b * m / S

    # empirically we find timing 0.005 tend to better performance
    loss_sum =  0.005 * (loss_sem + loss_pos)
    loss_sem_sum =  0.005 * loss_sem
    loss_pos_sum = 0.005 * loss_pos

    return loss_sum, loss_sem_sum,  loss_pos_sum
