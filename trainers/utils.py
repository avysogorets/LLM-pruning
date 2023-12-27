from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
import torch
import logging
import torch.nn.functional as F


def should_early_stop(metrics, patience):
    losses = {epoch: metrics[epoch]['loss'] for epoch in metrics.keys()}
    losses = [losses[i] for i in sorted(losses.keys())]
    if len(losses)>patience and losses[-patience]<losses[-1]:
        return True
    return False


def compute_metrics(y_true, outs, aggregation='macro'):
    y_pred = torch.argmax(outs, dim=1)
    metrics = precision_recall_fscore_support(
        y_true.cpu().numpy().reshape(-1), 
        y_pred.cpu().numpy().reshape(-1), 
        average=aggregation)
    metrics = {
        'precision': metrics[0],
        'recall': metrics[1],
        'f1_score': metrics[2],
        'loss': F.cross_entropy(outs, y_true).item(),
        'accuracy': (y_true == y_pred).float().mean().item(),
        'collapse': len(Counter(y_pred.cpu().numpy()))==1}
    return metrics


def get_train_information(trainer, epoch):
    epoch_str = '0'*(len(str(trainer.num_epochs))-len(str(epoch+1)))+str(epoch+1)
    lrs = []
    for param_group in trainer.optimizer.param_groups:
        lrs.append(str(param_group['lr']))
    lrs = '/'.join(lrs)
    information = (f"[epoch: {epoch_str}/{trainer.num_epochs}]"
                    f"[lr: {lrs}]"
                    f"[train loss/acc: {trainer.train_metrics[epoch]['loss']:.4f}/"
                    f"{trainer.train_metrics[epoch]['accuracy']:.4f}]"
                    f"[val loss/acc: {trainer.val_metrics[epoch]['loss']:.4f}/"
                    f"{trainer.val_metrics[epoch]['accuracy']:.4f}]"
                    f"[collapse: {trainer.val_metrics[epoch]['collapse']}]")
    return information