from ..utils import get_all_subclasses
from ..utils import layer_sparsity
from ..utils import model_sparsity
from .pruner_base import PrunerBase
from .dense import Dense
from .uniform import RandomUniform
from .igq import RandomIGQ
from .magnitude import Magnitude
from .synflow import SynFlow
from .snip import SNIP
from .grasp import GraSP
from .utils import *

def PrunerFactory(method_name, **kwargs):
    implemented_pruners = {}
    for _class_ in get_all_subclasses(PrunerBase):
        implemented_pruners[_class_.__name__] = _class_
    if method_name in implemented_pruners:
        return implemented_pruners[method_name](**kwargs)
    else:
        raise NotImplementedError(f"pruner {method_name} is unknown")