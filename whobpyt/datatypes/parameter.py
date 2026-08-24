"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""


import torch
import numpy
import numpy as np

class par:
    '''
    Features of this class:
     - This class contains a global parameter value or array of values (one per node)
     - It can also contain associated priors (mean and variance)
     - It also has attributes for whether the parameter and/or priors should be fit during training
     - It has a method to return the parameter as numpy value
     - It has a method to set a random val using based on priors
     - It has functionality to represent val as log(val) so that during training val will be constrained to be positive

    Attributes
    ------------
    val : Tensor
        The parameter value (or an array of node specific parameter values)
    prior_mean : Tensor
        Prior mean of the data value
    prior_precision : Tensor
        Prior inverse of variance of the value
    
    fit_par: Bool
        Whether the parameter value should be set to as a PyTorch Parameter
    fit_hyper : Bool
        Whether the parameter prior mean and prior variance should be set as a PyTorch Parameter
    asLog : Bool
        Whether the log of the parameter value will be stored instead of the parameter itself (will prevent parameter from being negative).
    '''

    def __init__(self, val, prior_mean = None, prior_std = None, fit_par = False, asLog = False, asRand = True, lb = 0, device = torch.device('cpu'), asLogit = False, ub = 1.0, l1_weight = 0.0):
        '''

        Parameters
        ----------
        val : Float (or Array)
            The parameter value (or an array of node specific parameter values)
        prior_mean : Float
            Prior mean of the data value
        prior_std : Float
            Prior std of the value
        fit_par: Bool
            Whether the parameter value should be set to as a PyTorch Parameter
        device: torch.device
            Whether to run on CPU or GPU
        asLogit : Bool
            Whether the parameter value represents a logit, so that value() is bounded to (lb, ub)
            via a sigmoid, instead of being stored directly (or as log(val), see asLog).
        ub : Float
            Upper bound used when asLogit is True. Ignored otherwise.
        l1_weight : Float
            If > 0, AbstractLoss.l1_loss() will add l1_weight * sum(|value()|) to the objective
            for this parameter -- a sparsity-inducing penalty pulling value() toward lb (its
            zero-deviation point) that, unlike a ridge/precision-based prior, snaps
            fit-irrelevant entries to exactly that point rather than merely shrinking them.
            Most useful for large (e.g. per-edge connectivity-gain) matrix parameters, where a
            jointly-trained prior precision (fit_hyper) is degenerate (see prior_loss()'s
            docstring). Independent of, and stackable with, prior_mean/prior_std/fit_hyper.
        '''
        self.fit_par = fit_par
        self.device = device
        self.asLog = asLog
        self.asLogit = asLogit
        self.asRand = asRand
        self.lb = torch.tensor(lb, dtype=torch.float32).to(self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(self.device)
        self.l1_weight = l1_weight
        self.fit_hyper = False

        if self.fit_par:
            if np.all(prior_mean != None) & np.all(prior_std != None):

                prior_mean_ts = torch.tensor(prior_mean, dtype=torch.float32).to(self.device)
                self.prior_mean = prior_mean_ts
                prior_std_ts = torch.tensor(prior_std, dtype=torch.float32).to(self.device)
                self.prior_precision = 1/prior_std_ts**2
                if self.asRand == True:
                    if type(val) is np.ndarray:
                        val = prior_mean + prior_std * torch.randn_like(torch.tensor(val, dtype=torch.float32)).detach().numpy()
                    else:
                        val = prior_mean + prior_std * np.random.randn(1,)
                
                    
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts
                self.fit_hyper = True

            else:
                val_ts = torch.tensor(val, dtype=torch.float32).to(self.device)
                self.val = val_ts
        else:
            self.val = torch.tensor(val, dtype=torch.float32).to(self.device)




    def to(self, device):
        '''
        '''
        self.device = device
        
        self.val = self.val.to(self.device)


    def npValue(self):
        '''
        Returns
        --------
        NumPy of Value
            The parameter value(s) as a NumPy Array
        '''
        
        if self.asLog:
            return self.lb.detach().clone().cpu().numpy() + np.exp(self.val.detach().clone().cpu().numpy())
        elif self.asLogit:
            lb = self.lb.detach().clone().cpu().numpy()
            ub = self.ub.detach().clone().cpu().numpy()
            v = self.val.detach().clone().cpu().numpy()
            return lb + (ub - lb) / (1 + np.exp(-v))
        else:
            return self.lb.detach().clone().cpu().numpy() + self.val.detach().clone().cpu().numpy()

    def value(self):
        '''
        Returns
        ---------
        Tensor of Value
            The parameter value(s) as a PyTorch Tensor
        '''


        if self.asLog:
            return self.lb + torch.exp(self.val)
        elif self.asLogit:
            return self.lb + (self.ub - self.lb) * torch.sigmoid(self.val)
        else:
            return self.lb + self.val


    


