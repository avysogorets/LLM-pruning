from .pruner_base import PrunerBase
from .utils import score_pruning
from torch.utils.data import Subset
import torch
import numpy as np


class GraSP(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()
        
    def prune(self, model, data, target_sparsity, pruning_type, sample_size, **kwargs):
        weights = []
        for module in model.get_prunable_modules():
            weights.append(module.weight.data.detach())
        sample_idxs = np.random.choice(range(len(data.datasets['train'])),
                size=sample_size,
                replace=False)
        sample = Subset(data.datasets['train'], sample_idxs)
        hgp = self._hessian_gradient_product(model, sample)
        scores = [weight*hg for weight,hg in zip(weights, hgp)]
        masks = score_pruning(target_sparsity, scores, pruning_type)
        return masks
        
        
    def _hessian_gradient_product(self, model, dataset):
        grad = model.gradients(dataset, create_graph=True)
        grad = torch.cat([g.reshape(-1) for g in grad if g is not None])
        stop_grad = grad.detach()
        g = grad @ stop_grad
        parameters = []
        for module in model.get_prunable_modules():
            parameters.append(module.weight)
        hgp = torch.autograd.grad(g, parameters)
        hgp = [hg.detach() for hg in hgp]
        return hgp