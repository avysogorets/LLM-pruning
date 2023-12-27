from .utils import random_pruning
from .pruner_base import PrunerBase
import torch


class RandomUniform(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()

    def prune(self, model, target_sparsity, pruning_type, **kwargs):
        shapes = []
        for module in model.get_prunable_modules():
            shape = list(module.weight.data.cpu().detach().numpy().shape)
            shapes.append(torch.Tensor(shape))
        sparsities = self.quotas(target_sparsity, shapes)
        return random_pruning(sparsities, shapes, pruning_type)
    
    def quotas(self, target_sparsity, shapes):
        return [target_sparsity]*len(shapes)