from ..utils import get_all_subclasses
from .bert_classifier import ClassifierBERT
from .model_base import ClassificationModelBase

def ModelFactory(model_name, **kwargs):
    implemented_models = {}
    for _class_ in get_all_subclasses(ClassificationModelBase):
        implemented_models[_class_.__name__] = _class_
    if model_name in implemented_models:
        return implemented_models[model_name](**kwargs)
    else:
        raise NotImplementedError(f"model {model_name} is unknown")