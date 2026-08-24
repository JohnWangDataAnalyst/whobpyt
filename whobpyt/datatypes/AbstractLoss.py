"""
Authors: Andrew Clappison, John Griffiths, Zheng Wang, Davide Momi, Sorenza Bastiaens, Parsa Oveisi, Kevin Kadak, Taha Morshedzadeh, Shreyas Harita
"""

import torch
from whobpyt.datatypes.parameter import par

class AbstractLoss:
    # This is the abstract class for objective function components, or for a custom objective function with multiple components. 

    def __init__(self, simKey = None, model = None, device = torch.device('cpu')):
    
        self.simKey = simKey #This is a string key to extract from the dictionary of simulation outputs the time series used by the objective function
        self.device =  device
        self.model = model
    
    def main_loss(self, simData, empData):
        # Calculates a loss to be backpropagated through
        # If the objective function needs additional info, it should be defined at initialization so that the parameter fitting paradigms don't need to change
        
        # simData: is a dictionary of simulated state variable/neuroimaging modality time series. Typically accessed as simData[self.simKey].
        # empData: is the target either as a time series or a calculated phenomena metric
        
        pass
    
    def prior_loss(self):
        loss_prior = []
        lb =0.001
        m = torch.nn.ReLU()
        variables_p = [a for a in dir(self.model.params) if type(getattr(self.model.params, a)) == par]
        # get penalty on each model parameters due to prior distribution
        for var_name in variables_p:
            # print(var)
            var = getattr(self.model.params, var_name)
            if var.fit_hyper:
                # NOTE: precision (m(var.prior_precision)) is ReLU-wrapped because a precision
                # must stay non-negative. val/prior_mean must NOT be -- they previously were
                # (m(var.val) - m(var.prior_mean)), which silently zeroes this term's gradient
                # on val whenever val<0 (relu(negative)==0 identically). That's a real cost for
                # any parameter using asLog with value<1 (val=log(value)<0) -- e.g. a noise std
                # around 0.1-0.2 -- which then trains completely unregularized by its own prior,
                # free to wander (observed: collapsing to a degenerate near-zero-noise solution)
                # regardless of how tight prior_std was set.
                loss_prior.append(torch.sum((lb + m(var.prior_precision)) * \
                                                (var.val - var.prior_mean) ** 2) \
                                      + torch.sum(-torch.log(lb + m(var.prior_precision)))) #TODO: Double check about converting _v_inv
        return loss_prior

    def l1_loss(self):
        """L1 (sparsity-inducing) penalty for any `par` with l1_weight > 0 -- e.g. a per-edge
        connectivity-gain matrix (RWW's w_ll; JR's w_ll/w_ff/w_bb) where a jointly-trained
        ridge precision (prior_loss's fit_hyper path) is degenerate (gradient descent finds it
        cheapest to just drive precision toward 0 rather than pay a penalty that scales with
        element count). Unlike a ridge, which pulls every entry diffusely toward the prior,
        L1's non-smooth-at-zero penalty snaps fit-irrelevant entries to exactly value()'s
        zero-deviation point (value()==lb) and reproducibly keeps them there.
        """
        loss_l1 = []
        variables_p = [a for a in dir(self.model.params) if type(getattr(self.model.params, a)) == par]
        for var_name in variables_p:
            var = getattr(self.model.params, var_name)
            if getattr(var, "l1_weight", 0.0) > 0:
                loss_l1.append(var.l1_weight * torch.sum(torch.abs(var.value() - var.lb)))
        return loss_l1
