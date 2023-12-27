from .utils import score_pruning
from .pruner_base import PrunerBase
from torch.utils.data import Subset
import numpy as np
import torch


class SNIP(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()
    
    def prune(self, model, data, target_sparsity, pruning_type, sample_size, **kwargs):
        model.eval()
        weights = []
        for module in model.get_prunable_modules():
            weights.append(module.weight.data.detach())
        sample_idxs = np.random.choice(range(len(data.datasets['train'])),
                size=sample_size,
                replace=False)
        sample = Subset(data.datasets['train'], sample_idxs)
        gradients = model.gradients(sample)
        scores = [torch.abs(gradient*weight) for gradient,weight in zip(gradients, weights)]
        masks = score_pruning(target_sparsity, scores, pruning_type)
        return masks