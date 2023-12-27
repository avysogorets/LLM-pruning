from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from .utils import compute_metrics, should_early_stop, get_train_information
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch
import logging

class Trainer:
    def __init__(self,
            model,
            data,
            lrs,
            weight_decay,
            iterations,
            batch_size,
            early_stopping,
            patience):

        self.model = model
        self.data = data
        self.iterations = iterations
        self.early_stopping = early_stopping
        self.patience = patience
        self.optimizer = torch.optim.Adam(
                [{'params': model.module_dict['embeddings'].parameters(),
                        'lr': lrs['embeddings']},
                {'params': model.module_dict['encoder'].parameters(),
                        'lr': lrs['encoder']},
                {'params': model.module_dict['classifier'].parameters(),
                        'lr': lrs['classifier'],
                        'weight_decay': weight_decay}],
                lr=lrs['classifier'])
        train_length = len(self.data.datasets['train'])
        lr_drops = [0.5*iterations, 0.75*iterations]
        drops = [int(drop*batch_size/train_length) for drop in lr_drops]
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=drops, gamma=0.5)
        self.num_epochs = int(self.iterations*batch_size/train_length)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        self.logger.info('[training starts]')
        train_dataloader = DataLoader(self.data.datasets['train'],
                batch_size=self.batch_size,
                shuffle=True)
        self.train_metrics = {}
        self.val_metrics = {}
        for epoch in range(self.num_epochs):
            epoch_outs = []
            epoch_y = []
            for X, y in tqdm(train_dataloader):
                self.model.train()
                self.optimizer.zero_grad()
                outs = self.model(X)
                loss = F.cross_entropy(outs, y)
                epoch_outs.append(outs)
                epoch_y.append(y)
                loss.backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            epoch_outs = torch.vstack(epoch_outs)
            epoch_y = torch.cat(epoch_y)
            self.train_metrics[epoch] = compute_metrics(epoch_y, epoch_outs)
            self.val_metrics[epoch] = self.validate()
            train_information = get_train_information(self, epoch)
            self.logger.info(train_information)
            if self.early_stopping and should_early_stop(self.val_metrics, self.patience):
                self.logger.info('[training finished (early stopping)]')
                return self.train_metrics, self.val_metrics
        self.logger.info('[training finished]')
        return self.train_metrics, self.val_metrics

    def validate(self):
        val_dataloader = DataLoader(self.data.datasets['dev'],
                batch_size=self.batch_size,
                shuffle=False)
        epoch_outs = []
        epoch_y = []
        self.model.eval()
        with torch.no_grad():
            for X,y in val_dataloader:
                outs = self.model(X)
                epoch_outs.append(outs)
                epoch_y.append(y)
        epoch_outs = torch.vstack(epoch_outs)
        epoch_y = torch.cat(epoch_y)
        val_metrics = compute_metrics(epoch_y, epoch_outs)
        return val_metrics
        



