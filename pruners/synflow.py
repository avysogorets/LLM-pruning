from .utils import score_pruning
from .pruner_base import PrunerBase
import logging
import torch


class SynFlow(PrunerBase):
    def __init__(self, num_iters=100, **kwargs):
        super().__init__()
        self.num_iters = num_iters
        self.logger = logging.getLogger(__name__)
        
    def get_scores(self, model):
        linearized = model.linearize()
        weights = []
        for module in model.get_prunable_modules():
            weights.append(module.weight.data.detach().cpu())
        X = torch.ones((512,linearized.input_length))
        X = X.to(model.device)
        output = torch.sum(linearized(X))
        output.backward()
        scores = []
        for module in linearized.get_prunable_modules():
            score = torch.abs(torch.mul(module.weight.grad, module.weight.data))
            scores.append(score)
        del linearized
        return scores
    
    def prune(self, model, target_sparsity, **kwargs):
        for iteration in range(self.num_iters):
            target_s = self.prune_schedule(iteration, target_sparsity)
            if iteration%10==0:
                self.logger.info(f'[SynFlow iteration {iteration}][target: {target_s:.7f}]')
            scores = self.get_scores(model)
            masks = score_pruning(target_s, scores)
            for i,module in enumerate(model.get_prunable_modules()):
                if isinstance(module, torch.nn.Embedding):
                    masks[i] = masks[i].t()
            model.update_masks(masks)
            model.apply_masks()
        return masks
    
    def prune_schedule(self, iteration, target_sparsity):
        target_d = (1-target_sparsity)**(iteration/(self.num_iters-1))
        return 1-target_d