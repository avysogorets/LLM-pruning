from torch.utils.data import Dataset
from typing import Dict

class ClassificationDataBase:
    """ Data base class API: loads, preprocesses, and tokenizes datasets to 
        be ready for the BERT model.
    """
    def __init__(self, **kwargs):
        self.num_classes: int
        self.datasets: Dict[str, Dataset] = {'train': None, 'dev': None}

