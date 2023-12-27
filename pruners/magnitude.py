from .utils import score_pruning
from .pruner_base import PrunerBase
import torch


class Magnitude(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()

    def prune(self, model, target_sparsity, pruning_type, **kwargs):
        if pruning_type=='direct':
            scores = self.scores(model)
            return score_pruning(target_sparsity, scores)
        
    def scores(self, model):
        scores = []
        for module in model.get_prunable_modules():
            magnitudes = torch.abs(module.weight.data.detach())
            scores.append(magnitudes)
        return scores