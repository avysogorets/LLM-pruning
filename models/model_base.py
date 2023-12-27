from typing import List, Dict, Set
from torch.utils.data import Dataset
import torch


class ClassificationModelBase(torch.nn.Module):
    """ The base model API
        self.module_dict is a dict of three torch.nn.Modules:
         - embeddings;
         - encoder;
         - classifier.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.masks: List[torch.Tensor]
        self.module_dict: torch.nn.ModuleDict
        self.prunable_modules: Set[torch.nn.Module]
    
    @property
    def effective_masks(self) -> List[torch.Tensor]:
        """ Compute effective masks corresponding to self.masks
        """
        raise NotImplementedError("override 'effective_masks' property")

    def _create_masks(self) -> None:
        """ Define self.masks for the unpruned model
        """
        raise NotImplementedError("override 'create_masks' method")

    def update_masks(self, masks) -> None:
        """ Update self.masks with masks
        """
        raise NotImplementedError("override 'update_masks' method")

    def apply_masks(self) -> None:
        """ Apply self.masks to the model
        """
        raise NotImplementedError("override 'apply_masks' method")

    def get_prunable_modules(self) -> List[torch.nn.Module]:
        """ Return a list of prunable modules (so that self.masks[i]
            applies to self.get_prunable_modules[i]).
        """
        raise NotImplementedError("override 'effective_masks' property")

    def forward(self, X: Dict) -> torch.Tensor:
        """ Forward pass of the model, return logits
        """
        raise NotImplementedError(f"override 'forward' method")

    def gradients(self, dataset: Dataset, create_graph: bool) -> List[torch.Tensor]:
        """ Compute gradients for weights of prunable modules, i.e.,
            gradients[i] corresponds to gradients of cross entropy
            with respect to self.get_prunbale_modules[i].weight.grad
            Set create_graph=True to retain graph and be able to compute
            higher-order derivatives.
        """
        raise NotImplementedError(f"override 'gradients' method")

    def linearize(self, **kwargs) -> torch.nn.Module:
        """ Return a linearized version of the model (i.e., strip
            away any non-lineaities, layernorms, batchnorms, etc.)
        """
        raise NotImplementedError(f"override 'linearize' method")


class LinearizedClassificationModelBase(torch.nn.Module):
    """ Linearized classification model API
    """
    def __init__(self):
        super().__init__()
        self.input_length: int
        self.module_dict: torch.nn.ModuleDict
    
    def forward(self, X: Dict) -> torch.Tensor:
        """ Forward pass of the model, return logits
        """
        raise NotImplementedError(f"override 'forward' method")

    def predict(self, dataset: Dataset) -> torch.Tensor:
        """ Forward pass of the model, return logits
        """
        raise NotImplementedError(f"override 'predict' method")

    def get_prunable_modules(self) -> List[torch.nn.Module]:
        """ Return a list of prunable modules (so that self.masks[i]
            applies to self.get_prunable_modules[i]).
        """
        raise NotImplementedError("override 'effective_masks' property")