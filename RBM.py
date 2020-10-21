#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 12:43:51 2020

@author: jamesnj1
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import sys


class RBM(nn.Module):
    
    def __init__(self,
                    visible_units   = 256,
                    hidden_units    = 64,
                    epsilonw        = 0.1, #learning rate for weights
                    epsilonvb       = 0.1, #learning rate for visible unit biases
                    epsilonhb       = 0.1, #learning rate for hidden unit biases
                    weightcost      = 0.0002,
                    initialmomentum = 0.5,
                    finalmomentum   = 0.9,
                    use_gpu = False
                    ):
        
        super(RBM,self).__init__()
        
        self.desc               = "RBM"
        self.visible_units      = visible_units
        self.hidden_units       = hidden_units
        self.epsilonw           = epsilonw
        self.epsilonvb          = epsilonvb
        self.epsilonhb          = epsilonhb
        self.weightcost         = weightcost
        self.initialmomentum    = initialmomentum
        self.finalmomentum      = finalmomentum
        self.use_gpu            = use_gpu
        self.device             = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.h_bias = nn.Parameter(torch.zeros(self.hidden_units)) #hidden layer bias
        self.v_bias = nn.Parameter(torch.zeros(self.visible_units)) #visible layer bias
        
        self.W = nn.Parameter(0.1*torch.randn(self.visible_units, self.hidden_units))
        
        if self.use_gpu:
            self.W = self.W.cuda()
            self.v_bias = self.v_bias.cuda()
            self.h_bias = self.h_bias.cuda()
            
            
    def positive_phase(self, X):
        '''
        Converts the data in visible layer to hidden layer
        X here is the visible probabilities
        :param X: torch tensor shape = (n_samples , n_features)
        :return -  X_prob - new hidden layer (probabilities)
                    sample_X_prob - Gibbs sampling of hidden (1 or 0) based
                                on the value
        '''
        
        X_prob  = torch.sigmoid(torch.add(torch.matmul(X, self.W), 
                                          self.h_bias))
        
        poshidstates    = self.sampling(X_prob)
        posprods        = torch.matmul(X.t(), X_prob)
        poshidact       = torch.sum(X_prob, 0)
        posvisact       = torch.sum(X, 0)
        
        return poshidstates, posprods, poshidact, posvisact
    
    
    def negative_phase(self, X):
        '''
        reconstructs data from hidden layer
        also does sampling
        X here is the probabilities in the hidden layer
        :returns - X_prob - the new reconstructed layers(probabilities)
                    sample_X_prob - sample of new layer(Gibbs Sampling)

        '''
        # computing hidden activations and then converting into probabilities
        negdata     = torch.sigmoid(torch.add(torch.matmul(X, self.W.transpose(0, 1)),
                                              self.v_bias))
        neghidprobs = torch.sigmoid(torch.add(torch.matmul(negdata, self.W), 
                                              self.h_bias))
        negprods    = torch.matmul(negdata.transpose(0, 1), neghidprobs)
        neghidact   = torch.sum(neghidprobs, 0)
        negvisact   = torch.sum(negdata,0)

        #sample_X_prob = self.sampling(negdata)

        return neghidprobs, negprods, neghidact, negvisact, negdata
    
    
    def adjust_weights(self, vishidinc, visbiasinc, hidbiasinc, posprods, negprods, 
                       posvisact, negvisact, poshidact, neghidact, epoch, num_epochs):
        
        numcases = self.batch_size
        
        if epoch >= math.ceil(num_epochs/2):
            momentum = self.finalmomentum
        else:
            momentum = self.initialmomentum
        
        vishidinc   = torch.mul(momentum, vishidinc) + torch.mul(self.epsilonw,
                                                                 torch.add(torch.div(torch.add(posprods, -negprods), numcases),
                                                                         -torch.mul(self.weightcost, self.W)))
        visbiasinc  = torch.mul(momentum, visbiasinc) + torch.mul(self.epsilonvb/numcases, torch.add(posvisact, -negvisact))
        hidbiasinc  = torch.mul(momentum, hidbiasinc) + torch.mul(self.epsilonhb/numcases, torch.add(poshidact, -neghidact))
        
        
        with torch.no_grad():
            self.W      += vishidinc
            self.v_bias += visbiasinc
            self.h_bias += hidbiasinc
    
        return vishidinc, visbiasinc, hidbiasinc
    
    
    def sampling(self,prob):
        '''
        Bernoulli sampling done based on probabilities s
        '''
        # s = torch.distributions.Bernoulli(prob).sample()
        s = prob > torch.rand(prob.size())
        return s.float()
    
    
    def forward(self,X):
        'data->hidden'
        if self.use_gpu:
            X = X.to(self.device)  
        out = torch.sigmoid(torch.add(torch.matmul(X, self.W), 
                                      self.h_bias))
        out_sampling = self.sampling(out)
        return out, out_sampling
    
    
    def backward(self, X):
        if self.use_gpu:
            X = X.to(self.device)
        out = torch.sigmoid(torch.add(torch.matmul(X, self.W.t()),
                                      self.v_bias))
        out_sampling = self.sampling(out)
        return out, out_sampling
    
    
    def compress_data_layer(self, X):
        if self.use_gpu:
            X = X.to(self.device)
        out = torch.add(torch.matmul(X, self.W), self.h_bias)
        return out
    
    
    def reconstruct(self, X):
         _, f_s = self.forward(X)
         b, _ = self.backward(f_s)
         return b
     
        
    def reconstruction_error(self, input_data):
         output = self.reconstruct(input_data)
         loss = nn.BCEWithLogitsLoss()(output, input_data)
         return loss
        
    
    def pass_through(self, data, epoch, num_epochs, vishidinc, hidbiasinc, visbiasinc):
        
        poshidstates, posprods, poshidact, posvisact = self.positive_phase(data)
        
        neghidprobs, negprods, neghidact, negvisact, negdata = self.negative_phase(poshidstates)
        
        err = torch.sum(torch.pow(torch.add(data, -negdata), 2))
            
        vishidinc, visbiasinc, hidbiasinc = self.adjust_weights(vishidinc, visbiasinc, hidbiasinc, posprods, negprods, 
                                                                posvisact, negvisact, poshidact, neghidact, epoch, num_epochs)
        
        return err, vishidinc, visbiasinc, hidbiasinc
    
    
    def pre_train(self, train_dataloader, num_epochs):
        self.batch_size = train_dataloader.batch_size
        n_batches = int(len(train_dataloader))
        
        vishidinc   = torch.zeros((self.visible_units, self.hidden_units))
        hidbiasinc  = torch.zeros(self.hidden_units)
        visbiasinc  = torch.zeros(self.visible_units)
        if self.use_gpu:
            vishidinc   = vishidinc.cuda()
            hidbiasinc  = hidbiasinc.cuda()
            visbiasinc  = visbiasinc.cuda()
        
        for epoch in range(1, num_epochs+1):
            error_ = torch.FloatTensor(n_batches , 1)
            for i,(batch,_) in tqdm(enumerate(train_dataloader), ascii=True, desc="RBM pre-train", file=sys.stdout): 
                batch = batch.view(len(batch) , self.visible_units)
                if self.use_gpu:
                    batch = batch.to(self.device)

                error_[i-1], vishidinc, visbiasinc, hidbiasinc = self.pass_through(batch, epoch, num_epochs, 
                                                                                   vishidinc, hidbiasinc, visbiasinc)
                
            print("Epoch {} error = {} ".format(epoch, torch.sum(error_)))
            
        return


