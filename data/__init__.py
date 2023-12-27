from ..utils import get_all_subclasses
from .data_base import ClassificationDataBase
from .nlp_data import IMDb, CoLa, SST2, AG_News, QNLI


def DataFactory(dataset_name, **kwargs):
    implemented_datasets = {}
    for _class_ in get_all_subclasses(ClassificationDataBase):
        implemented_datasets[_class_.__name__] = _class_
    if dataset_name in implemented_datasets:
        return implemented_datasets[dataset_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined dataset {dataset_name}")