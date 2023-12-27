from ..models.model_base import ClassificationModelBase
from torch import Tensor
from typing import List


class PrunerBase:
    """ The base pruner API
    """
    def prune(self, model: ClassificationModelBase,
                    target_sparsity: float,
                    pruning_type: str = 'direct',
                    **kwargs) -> List[Tensor]:
        """ Return a list of binary masks as result of pruning the model
        """
        raise NotImplementedError("override 'prune' method")