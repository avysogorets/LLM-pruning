from typing import List
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pandas as pd
import os


class TokenizedDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y)
            
    def __getitem__(self,idx):
        return {key: self.X[key][idx] for key in self.X.keys()}, self.y[idx]


def load_dataframes(loading_kwargs):
    path = loading_kwargs['path']
    train_filename = loading_kwargs['train_filename']
    dev_filename = loading_kwargs['dev_filename']
    header = loading_kwargs['header']
    index_col = loading_kwargs['index_col']
    assert '.' in train_filename, f"unrecognized file format {train_filename}"
    extension = train_filename.split('.')[-1]
    if extension == 'tsv':
        delimiter = '\t'
    elif extension == 'csv':
        delimiter = ','
    else:
        raise ValueError(f"unrecognized file format {extension}")
    dataframes = {}
    for split,split_filename in zip(['train','dev'], [train_filename, dev_filename]):
        filename = os.path.join(path, split_filename)
        dataframes[split] = pd.read_csv(
                filename,
                delimiter=delimiter,
                header=header,
                index_col=index_col,
                engine="python",
                on_bad_lines='skip')
        new_columns = [f"sentence_{i}" for i in range(len(dataframes[split].columns)-1)]+["label"]
        dataframes[split].columns = new_columns
    return dataframes


def get_tokens(data_X: List[List[str]], backbone_name: str) -> dict:
    tokenizer =  BertTokenizer.from_pretrained(backbone_name)
    if len(data_X[0])==1:
        data_X = [X[0] for X in data_X]
    data_X = tokenizer.batch_encode_plus(
                data_X,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt")
    return data_X