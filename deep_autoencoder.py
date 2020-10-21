#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 16:20:18 2020

@author: jamesnj1
"""

import torch, math, sys, itertools
import torch.nn as nn
from RBM import RBM
from tqdm import tqdm


class DBN(nn.Module):
    def __init__(self,
                 visible_units      = 256,
                 hidden_units       = [128, 64],
                 epsilonw           = 0.1, #learning rate for weights
                 epsilonvb          = 0.1, #learning rate for visible unit biases
                 epsilonhb          = 0.1, #learning rate for hidden unit biases
                 weightcost         = 0.0002,
                 initialmomentum    = 0.5,
                 finalmomentum      = 0.9,
                 batch_size         = 100,
                 use_gpu = False
                 ):
        super(DBN,self).__init__()
        
        self.n_layers   = len(hidden_units)
        self.batch_size = batch_size
        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rbm_layers = []
        
        # Creating different RBM layers
        for i in range(self.n_layers ):
            input_size = 0
            if i==0:
                input_size = visible_units
            else:
                input_size = hidden_units[i-1]
            rbm = RBM(visible_units     = input_size,
                      hidden_units      = hidden_units[i],
                      epsilonw          = epsilonw,
                      epsilonvb         = epsilonvb,
                      epsilonhb         = epsilonhb,
                      weightcost        = weightcost,
                      initialmomentum   = initialmomentum,
                      finalmomentum     = finalmomentum,
                      use_gpu           = use_gpu)
            self.rbm_layers.append(rbm)
            
            
    def forward(self, data):
        '''
        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions

        Returns
        -------
        v : TYPE torch.Tensor
            forward pass though the DBN, producing a compressed size dataset
	    technically, this is halfway through the unrolled DBN
        '''
        v = data.to(self.device)
        L = len(self.rbm_layers)/2
        for ii in range(L):
            v = v.view((v.shape[0] , -1))
            if ii != (L-1):
                v, _ = self.rbm_layers[ii].forward(v)  
            else:
                v = self.rbm_layers[ii].compress_data_layer(v)
        return v
    
    
    def backward(self, compressed_data):
        '''
        Parameters
        ----------
        compressed_data : TYPE torch.Tensor
            N x D tensor, where D <= D* (from original dataset)
            this is the (compressed) data that is extracted from the
            code layer of the autoencoder

        Returns
        -------
        v : TYPE torch.Tensor
            N x D tensor, where D = D* (from original dataset)
            this passes the compressed data backward through the DBN
            to reconstruct it

        '''
        v = compressed_data.to(self.device)
        for ii in range(len(self.rbm_layers)-1, -1, -1):
            v = v.view(v.shape[0], -1)
            v, _ = self.rbm_layers[ii].backward(v)
        return v 
    
    
    def reconstruct(self, data):
        '''
        Produces a forward and backward pass through the NON-unrolled
        DBN
        
        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions

        Returns
        -------
        v : TYPE torch.Tensor
            N x D tensor of estimates for each entry in data

        '''
        v = self.backward(self.forward(data))
        return v
    
    
    def pass_through_full(self, data):
        '''
        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions

        Returns
        -------
        v : TYPE torch.Tensor
            N x D tensor of estimates for each entry in data,
            ONLY for an unrolled network

        '''
        v = data.to(self.device)
        L = len(self.rbm_layers)
        for ii in range(L):
            if ii < math.floor(L/2)-1:
                v, _ = self.rbm_layers[ii].forward(v)
            elif ii == math.floor(L/2)-1:
                v = self.rbm_layers[ii].compress_data_layer(v)
            else:
                v, _ = self.rbm_layers[ii].backward(v)
        
        return v
    
    
    def pretrain(self, data, labels, num_epochs=10):
        '''

        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions
        labels : TYPE torch.Tensor
            N x 1 vector of labels for each sample
        num_epochs : TYPE, optional
            Number of epochs to train each RBM layer. The default is 10.

        Returns
        -------
        None.

        '''
        #train visible layer first
        print("-"*20)
        print("Training RBM layer {}".format(1))
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            
        dataset = torch.utils.data.TensorDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.rbm_layers[0].pre_train(dataloader, num_epochs)
        p_v , _ = self.rbm_layers[0].forward(data)
        for ii in range(1,len(self.rbm_layers)):
            print("-"*20)
            print("Training RBM layer {}".format(ii+1))
            dataset = torch.utils.data.TensorDataset(p_v, labels)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            self.rbm_layers[ii].pre_train(dataloader, num_epochs)
            p_v , _ = self.rbm_layers[ii].forward(p_v)
            
        return
        
    
    def fine_tuning(self, data, labels, num_epochs=10, max_iter=3):
        '''
        Parameters
        ----------
        data : TYPE torch.Tensor
            N x D tensor with N = num samples, D = num dimensions
        labels : TYPE torch.Tensor
            N x 1 vector of labels for each sample
        num_epochs : TYPE, optional
            DESCRIPTION. The default is 10.
        max_iter : TYPE, optional
            DESCRIPTION. The default is 3.

        Returns
        -------
        None.

        '''
        N = data.shape[0]
        #need to unroll the weights into a typical autoencoder structure
        #encode - code - decode
        for ii in range(len(self.rbm_layers)-1, -1, -1):
            self.rbm_layers.append(self.rbm_layers[ii])
        
        L = len(self.rbm_layers)
        optimizer = torch.optim.LBFGS(params=list(itertools.chain(*[list(self.rbm_layers[ii].parameters()) 
                                                                    for ii in range(L)]
                                                                  )),
                                      max_iter=max_iter,
                                      line_search_fn='strong_wolfe') 
        
        dataset     = torch.utils.data.TensorDataset(data, labels)
        dataloader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size*10, shuffle=True)
        #fine tune weights for num_epochs
        for epoch in range(1,num_epochs+1):
            #get squared error before optimization
            v = self.pass_through_full(data)
            err = (1/N) * torch.sum(torch.pow(data-v, 2))
            print("Before epoch {}, train squared error: {:.4f}\n".format(epoch, err))
        
            for ii,(batch,_) in tqdm(enumerate(dataloader), ascii=True, desc="DBN fine-tuning", file=sys.stdout):
                print("Fine-tuning epoch {}, batch {}".format(epoch, ii))
                batch = batch.view(len(batch) , self.rbm_layers[0].visible_units).to(self.device)
                B = batch.shape[0]
                def closure():
                    optimizer.zero_grad()
                    output = self.pass_through_full(batch)
                    loss = nn.BCELoss(reduction='sum')(output, batch)/B
                    print("Batch {}, loss: {}\r".format(ii, loss))
                    loss.backward()
                    return loss
                optimizer.step(closure)
                
        return














