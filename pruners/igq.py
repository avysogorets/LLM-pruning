from .utils import random_pruning
from .pruner_base import PrunerBase
import torch


class RandomIGQ(PrunerBase):
    def __init__(self, **kwargs):
        super().__init__()

    def prune(self, model, target_sparsity, pruning_type, **kwargs):
        shapes = []
        for module in model.get_prunable_modules():
            shape = list(module.weight.data.cpu().detach().numpy().shape)
            shapes.append(torch.Tensor(shape))
        sparsities = self.quotas(target_sparsity, shapes)
        return random_pruning(sparsities, shapes, pruning_type=pruning_type)

    def _bs_force_igq(self, areas, Lengths, target_sparsity, tolerance,f_low,f_high, depth):
        lengths_low=[Length/(f_low/area+1) for Length,area in zip(Lengths,areas)]
        overall_sparsity_low=1-sum(lengths_low)/sum(Lengths)
        if abs(overall_sparsity_low-target_sparsity)<tolerance or depth<0:
            return [1-length/Length for length,Length in zip(lengths_low,Lengths)]
        lengths_high=[Length/(f_high/area+1) for Length,area in zip(Lengths,areas)]
        overall_sparsity_high=1-sum(lengths_high)/sum(Lengths)
        if abs(overall_sparsity_high-target_sparsity)<tolerance or depth<0:
            return [1-length/Length for length,Length in zip(lengths_high,Lengths)]
        force=float(f_low+f_high)/2
        lengths=[Length/(force/area+1) for Length,area in zip(Lengths,areas)]
        overall_sparsity=1-sum(lengths)/sum(Lengths)
        f_low=force if overall_sparsity<target_sparsity else f_low
        f_high=force if overall_sparsity>target_sparsity else f_high
        return self._bs_force_igq(areas,Lengths,target_sparsity,tolerance,f_low,f_high, depth-1)

    def quotas(self, target_sparsity, shapes):
        counts=[torch.prod(shape) for shape in shapes]
        tolerance=100./sum(counts)
        areas=[1./count for count in counts]
        Lengths=[count for count in counts]
        return self._bs_force_igq(areas,Lengths,target_sparsity,tolerance,0,1e30, 1000)