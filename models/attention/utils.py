import torch

def nanstd(o,dim):
    return torch.sqrt(torch.nanmean(torch.pow(torch.abs(o-torch.nanmean(o,dim=dim).unsqueeze(dim)),2),dim=dim))

