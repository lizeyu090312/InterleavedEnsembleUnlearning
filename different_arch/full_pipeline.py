import torch
from torch import nn


class FullPipeline_Ensemble(nn.Module):
    def __init__(self, robust_module, poison_module, **kwargs):
        super(FullPipeline_Ensemble, self).__init__()

        self.robust_module = robust_module
        self.poison_module = poison_module

    def forward(self, x, confidence_threshold, mode:str, x_no_crop=None, **kwargs):
        poison_inp = x_no_crop if x_no_crop is not None else x
        poison_out = self.poison_module(poison_inp)
        if mode == "prefinetuning":
            return {"poison_out": poison_out, "robust_out": None, "weight": None}
        
        robust_out = self.robust_module(x)
        
        # robust_out and poison_out should be (b,n_class) at this point
        poison_out_softmax_max = torch.abs(torch.amax(torch.softmax(poison_out, dim=-1), dim=-1, keepdim=True))
        weight = (poison_out_softmax_max < confidence_threshold).float()
        
        return {"poison_out": poison_out, "robust_out": robust_out, "weight": weight}
