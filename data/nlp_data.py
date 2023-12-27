from .utils import load_dataframes, get_tokens
from .utils import TokenizedDataset
from .data_base import ClassificationDataBase
import torch


class NLPData(ClassificationDataBase):
    def __init__(self, **kwargs):
        super().__init__()
    
    def _prepare_datasets(self, loading_kwargs, device, backbone_name):
        dataframes = load_dataframes(loading_kwargs)
        self.num_classes = len(dataframes['train']['label'].unique())
        data_X, data_y = {}, {}
        for split in dataframes.keys():
            data_X[split] = dataframes[split].drop('label', axis=1, inplace=False)
            data_X[split] = data_X[split].values.tolist()
            data_y[split] = dataframes[split]['label'].values
            data_y[split] = torch.LongTensor(data_y[split]).to(device)
            data_X[split] = get_tokens(data_X[split], backbone_name=backbone_name)
            data_X[split] = {k: v.to(device) for k,v in data_X[split].items()}
            self.datasets[split] = TokenizedDataset(data_X[split], data_y[split])


class IMDb(NLPData):
    def __init__(self, device, backbone_name, **kwargs):
        super().__init__()
        loading_kwargs = {
                'path': '/scratch/amv458/nlp/datasets/imdb', 
                'train_filename': 'train.tsv',
                'dev_filename': 'dev.tsv',
                'header': 0,
                'index_col': 0}
        self.device = device
        self._prepare_datasets(loading_kwargs, device, backbone_name)


class CoLa(NLPData):
    def __init__(self, device, backbone_name, **kwargs):
        super().__init__()
        loading_kwargs = {
                'path': '/scratch/amv458/nlp/datasets/cola', 
                'train_filename': 'train.tsv',
                'dev_filename': 'dev.tsv',
                'header': None,
                'index_col': None}
        self.device = device
        self._prepare_datasets(loading_kwargs, device, backbone_name)


class AG_News(NLPData):
    def __init__(self, device, backbone_name, **kwargs):
        loading_kwargs = {'path': '/scratch/amv458/nlp/datasets/ag_news',
                          'train_filename': 'train.csv',
                          'dev_filename': 'dev.csv',
                          'header': None,
                          'index_col': None}
        super().__init__(loading_kwargs=loading_kwargs, **kwargs)
        self.device = device
        self._prepare_datasets(loading_kwargs, device, backbone_name)


class SST2(NLPData):
    def __init__(self, device, backbone_name, **kwargs):
        loading_kwargs = {'path': '/scratch/amv458/nlp/datasets/sst2',
                          'train_filename': 'train.tsv',
                          'dev_filename': 'dev.tsv',
                          'header': 0,
                          'index_col': 0}
        super().__init__(loading_kwargs=loading_kwargs, **kwargs)
        self.device = device
        self._prepare_datasets(loading_kwargs, device, backbone_name)


class QNLI(NLPData):
    def __init__(self, device, backbone_name, **kwargs):
        loading_kwargs = {'path': '/scratch/amv458/nlp/datasets/qnli',
                          'train_filename': 'train.tsv',
                          'dev_filename': 'dev.tsv',
                          'header': 0,
                          'index_col': 0}
        super().__init__(loading_kwargs=loading_kwargs, **kwargs)
        self.device = device
        self._prepare_datasets(loading_kwargs, device, backbone_name)

