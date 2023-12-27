from .models import *
from .models.attention import *
from .data import *
from .pruners import *
from .trainers import *
from .utils import *
import torch
import argparse
import logging.config
import os

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument('--save', action='store_true', default=False, help='save output files')
parser.add_argument('--train', action='store_true', default=False, help='train or not')
parser.add_argument('--seed', type=int, default=0, help='global seed')
parser.add_argument('--result_path', type=str, default='/scratch/amv458/project/res', help='output path')
parser.add_argument('--log_path', type=str, default='/scratch/amv458/project/log', help='logs path')
parser.add_argument('--out_filename', type=str, default='metrics.json', help='output filename')

# data parameters
parser.add_argument('--path', type=str, default='/scratch/amv458/datasets', help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='IMDb', help='dataset name')

# model parameters
parser.add_argument('--backbone_name', type=str, default='bert-base-uncased', help='LLM to use as feature extractor')
parser.add_argument('--attention_name', type=str, default='BertSelfAttentionOriginal',help='BERT self-attention module')
parser.add_argument('--freeze_embeddings', action='store_true', default=False, help='freeze embeddings layers')
parser.add_argument('--prune_classifier', action='store_true', default=False, help='prune classifier')
parser.add_argument('--model_name', type=str, default='ClassifierBERT', help='')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate of the classifier')
parser.add_argument('--pool_type', type=str, default='mean', help='pooling type of LLM embeddings: avg, mean / cls, first / full / max')

# training parameters
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--lr_embedding', type=float, default=8e-7, help='learning rate of LLM embeddings')
parser.add_argument('--lr_encoder', type=float, default=8e-7, help='learning rate of LLM encoder')
parser.add_argument('--lr_classifier', type=float, default=8e-5, help='learning rate of classifier')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='L2 penalty coefficient')
parser.add_argument('--early_stopping', action='store_true', default=False, help='early stopping based on validation loss')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience (in validation epochs)')
parser.add_argument('--iterations', type=int, default=50000, help='training iterations')

# pruning parameters
parser.add_argument('--pruner_name', type=str, default='SNIP', help='pruner to use')
parser.add_argument('--pruning_type', type=str, default='direct', help='direct / effective pruning')
parser.add_argument('--sample_size', type=int, default=256, help='data sample size for SNIP and GraSP')
parser.add_argument('--compression', type=float, default=1, help='target compression')

args=parser.parse_args()
if args.freeze_embeddings:
        args.lr_embedding = 0
if args.pruner_name == 'Dense':
     args.compresssion = 1


def main(args):
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    fileid = get_fileid(
            args.seed,
            args.dataset_name,
            args.backbone_name,
            args.attention_name,
            args.model_name,
            args.pruner_name,
            args.pruning_type,
            args.prune_classifier,
            int(args.compression))
    log_filename = '.'.join([fileid, 'log'])
    log_filename = os.path.join(args.log_path, log_filename)
    logging.basicConfig(
            filename=log_filename,
            encoding='utf-8',
            filemode='w',
            level=logging.INFO)
    print(f'[logs saved to {log_filename}]')
    logger = logging.getLogger(__name__)
    specs_info = get_specs_info(args)
    logger.info(specs_info)
    set_seeds(args.seed)
    print('past specs')
    print(specs_info)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    pruner = PrunerFactory(args.pruner_name)
    data = DataFactory(
            dataset_name=args.dataset_name,
            backbone_name=args.backbone_name,
            device=device)
    attention_class = AttentionFactory(args.attention_name)
    model = ModelFactory(
            model_name=args.model_name,
            backbone_name=args.backbone_name,
            pool_type=args.pool_type,
            num_classes=data.num_classes,
            freeze_embeddings=args.freeze_embeddings,
            attention_class=attention_class,
            prune_classifier=args.prune_classifier,
            device=device)
    lrs = {'embeddings': args.lr_embedding,
           'encoder': args.lr_encoder,
           'classifier': args.lr_classifier}
    trainer = Trainer(
            model=model,
            data=data,
            lrs=lrs,
            weight_decay=args.weight_decay,
            iterations=args.iterations,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            patience=args.patience)
    target_sparsity = 1-1./(args.compression)
    masks = pruner.prune(
            model=model,
            data=data,
            target_sparsity=target_sparsity,
            pruning_type=args.pruning_type,
            sample_size=args.sample_size)
    model.update_masks(masks)
    init_metrics = collect_init_metrics(
            model,
            data.datasets["train"],
            data.num_classes,
            num_samples=32)
    direct_s = model_sparsity(masks)
    effective_s = model_sparsity(model.effective_masks)
    res_filename = '.'.join([fileid+'_init', 'json'])
    logger.info(f'[saving init results][filename: {res_filename}]')
    save(path=args.result_path, filename=res_filename, metrics=init_metrics)
    information = (f"[pruning completed]"
                   f"[pruner: {args.pruner_name}]"
                   f"[requested: {target_sparsity:.7f}]"
                   f"[direct: {direct_s:.7f}]"
                   f"[effective: {effective_s:.7f}]")
    logger.info(information)
    train_metrics, val_metrics = trainer.train()
    train_metrics = collect_train_metrics(model, train_metrics, val_metrics)
    res_filename = '.'.join([fileid+'_train', 'json'])
    logger.info(f'[saving train results][filename: {res_filename}]')
    save(path=args.result_path, filename=res_filename, metrics=train_metrics)


if __name__ == "__main__":
    main(args)