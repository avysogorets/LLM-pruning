from .pruner_base import PrunerBase
import torch


class Dense(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()

    def prune(self, model, **kwargs):
        masks = []
        for module in model.get_prunable_modules():
            mask = torch.ones(module.weight.data.size())
            masks.append(mask)
        return masks