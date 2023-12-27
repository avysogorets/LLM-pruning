from typing import List
import math
from datetime import datetime
from scipy.sparse.linalg import svds
from torch.utils.data import Subset, DataLoader
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import json
import torch
import os


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def layer_sparsity(mask: torch.Tensor):
    density = torch.sum(mask>0)/np.prod(mask.cpu().numpy().shape)
    return 1-density.item()

def get_specs_info(args):
    now = datetime.now()
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")
    info_msg = (
            f"\n\n[seed          ] {args.seed}\n"
            f"[freeze embs   ] {args.freeze_embeddings}\n"
            f"[train         ] {args.train}\n"
            f"[prune class   ] {args.prune_classifier}\n"
            f"[backbone name ] {args.backbone_name}\n"
            f"[attn name     ] {args.attention_name}\n"
            f"[dataset name  ] {args.dataset_name}\n"
            f"[pruner name   ] {args.pruner_name}\n"
            f"[compression   ] {args.compression}\n"
            f"[iterations    ] {args.iterations}\n"
            f"[batch size    ] {args.batch_size}\n"
            f"[lr embedding  ] {args.lr_embedding}\n"
            f"[lr encoder    ] {args.lr_encoder}\n"
            f"[lr classifier ] {args.lr_classifier}\n"
            f"[timestamp     ] {now_str}\n")
    return info_msg


def model_sparsity(masks: List[torch.Tensor]):
    counts = [np.prod(mask.cpu().numpy().shape) for mask in masks]
    sparsities = [layer_sparsity(mask) for mask in masks]
    active_parameters = 0
    for count,sparsity in zip(counts, sparsities):
        active_parameters+=count*(1-sparsity)
    return 1-active_parameters/sum(counts)


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def collect_train_metrics(model, train_metrics, val_metrics):
    metrics = {'train_metrics': train_metrics,
               'val_metrics': val_metrics}
    lds = {i: layer_sparsity(mask) for i,mask in enumerate(model.masks)}
    metrics['layer_direct_sparsity'] = lds
    les = {i: layer_sparsity(mask) for i,mask in enumerate(model.effective_masks)}
    metrics['layer_effective_sparsity'] = les
    metrics['model_direct_sparsity'] = model_sparsity(model.masks)
    metrics['model_effective_sparsity'] = model_sparsity(model.effective_masks)
    return metrics

def save(path, filename, metrics):
    dump_path_filename = os.path.join(path, filename)
    f = open(dump_path_filename, 'w')
    json.dump(metrics, f)
    f.close()


def get_fileid(*args):
    fileid = '_'.join([str(arg) for arg in args])
    return fileid

def NC1(model, dataset, num_classes):
    representations = [[] for k in range(num_classes)]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for X,y in dataloader:
        with torch.no_grad():
            representation = model(X, embeddings=True)
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    dim = representations[0].shape[1]
    SW = torch.zeros((dim, dim)).to(model.device)
    for y in range(num_classes):
        for representation in representations[y]:
            vec = (representation-mu_k[y]).reshape(-1,1)
            SW += torch.matmul(vec, vec.T)
    SW /= len(dataset)
    SG = torch.zeros((dim, dim)).to(model.device)
    for y in range(num_classes):
        vec = (mu_k[y]-mu_g).reshape(-1,1)
        SG += torch.matmul(vec, vec.T)
    SG = (SG/num_classes)
    eigvec, eigval, _ = svds(SG.cpu().numpy(), k=num_classes-1)
    inv_Sb = eigvec@np.diag(eigval**(-1))@eigvec.T 
    nc1 = np.trace(SW.cpu().numpy()@inv_Sb)/num_classes
    return nc1

def NC2(model, dataset, num_classes):
    representations = [[] for k in range(num_classes)]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for X,y in dataloader:
        with torch.no_grad():
            representation = model(X, embeddings=True)
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    M = []
    for k in range(num_classes):
        vec = (mu_k[k]-mu_g).reshape(1,-1)
        vec = vec / vec.norm()
        M.append(vec)
    M = torch.vstack(M)
    A = torch.matmul(M,M.T)/torch.norm(torch.matmul(M,M.T), p='fro')
    one_k = torch.ones((num_classes, 1))
    I_k = torch.eye(num_classes)
    const_1 = 1./np.sqrt(num_classes-1)
    const_2 = 1./num_classes
    factor = torch.matmul(one_k, one_k.T)
    nc2 = torch.norm(A.cpu()-const_1*(I_k-const_2*factor) , p='fro')
    return nc2.item()

def NC3(model, dataset, num_classes):
    representations = [[] for k in range(num_classes)]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for X,y in dataloader:
        with torch.no_grad():
            representation = model(X, embeddings=True)
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for k in range(num_classes):
        representations[k] = torch.vstack(representations[k])
    mu_k = [torch.mean(representations[k], dim=0) for k in range(num_classes)]
    mu_g = torch.mean(torch.vstack(mu_k), dim=0)
    M = []
    for k in range(num_classes):
        vec = (mu_k[k]-mu_g).reshape(1,-1)
        vec = vec / vec.norm()
        M.append(vec)
    M = torch.vstack(M)
    C = model.module_dict['classifier'].weight.data
    AM = torch.matmul(C, M.T)/torch.norm(torch.matmul(C, M.T), p='fro')
    one_k = torch.ones((num_classes, 1))
    I_k = torch.eye(num_classes)
    const_1 = 1./np.sqrt(num_classes-1)
    const_2 = 1./num_classes
    factor = torch.matmul(one_k, one_k.T)
    nc2 = torch.norm(AM.cpu()-const_1*(I_k-const_2*factor) , p='fro')
    return nc2.item()

def NC4(model, dataset, num_classes):
    representations = [[] for k in range(num_classes)]
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for X,y in dataloader:
        with torch.no_grad():
            representation = model(X, embeddings=True)
        if representation.isnan().any() or representation.isinf().any():
            return float('inf')
        representations[y].append(representation)
    for y in range(num_classes):
        representations[y] = torch.vstack(representations[y])
    mu_k = [torch.mean(representations[y], dim=0) for y in range(num_classes)]
    nn_score = 0
    for y in range(num_classes):
        for representation in representations[y]:
            true_class_norm = (representation-mu_k[y]).norm()
            nn_correct = True
            for c in range(num_classes):
                 if (representation-mu_k[c]).norm()<true_class_norm:
                    nn_correct = False
            nn_score += int(nn_correct)
    return 1-nn_score/len(dataset)


def get_entropy(model, dataset):
    model.eval()
    hooks_ks = []
    hooks_qs = []
    ks = []
    qs = []
    batch_size=4
    def k_hook(m, inp, out):
        ks.append(out.detach())
    def q_hook(m, inp, out):
        qs.append(out.detach())
    for layer_id in range(len(model.module_dict['encoder'].layer)):
        m = model.module_dict['encoder'].layer[layer_id]
        hooks_ks.append(m.attention.self.key.register_forward_hook(k_hook))
        hooks_qs.append(m.attention.self.query.register_forward_hook(q_hook))
    model.predict(dataset, batch_size=batch_size)
    kq_to_scores = model.module_dict['encoder'].layer[0].attention.self.kq_to_scores
    num_layers = model.get_config().num_hidden_layers
    num_heads = model.get_config().num_attention_heads
    attn_head_size = model.get_config().hidden_size//num_heads
    length = len(dataset[0][0]['input_ids'])
    shape = (num_layers,len(dataset),num_heads,length)
    entropies = np.zeros(shape)
    for reg_id in range(len(ks)):
        k = ks[reg_id]
        q = qs[reg_id]
        data_id = batch_size*(reg_id//num_layers)
        attention_mask = torch.vstack([dataset[i][0]['attention_mask'] for i in range(data_id,min(data_id+batch_size,len(dataset)))])
        attention_mask = attention_mask[:,None,None,:]
        attention_scores = kq_to_scores(k, q,
                attention_mask=attention_mask,
                attention_head_size=attn_head_size)
        attention_scores = attention_scores.squeeze()
        probs = F.softmax(attention_scores, dim=-1)
        ents = Categorical(probs).entropy().cpu().numpy()
        layer_id = reg_id%num_layers
        entropies[layer_id][data_id:data_id+batch_size] = ents
    entropies.transpose((1,0,2,3)).shape
    for hook_k in hooks_ks:
        hook_k.remove()
    for hook_q in hooks_qs:
        hook_q.remove()
    return entropies

def collect_init_metrics(
            model,
            dataset,
            num_classes,
            num_samples):
    idxs = np.random.choice(range(len(dataset)), size=num_samples, replace=False)
    subset = Subset(dataset, idxs)
    metrics = {
        'nc1': NC1(model, subset, num_classes),
        'nc2': NC2(model, subset, num_classes),
        'nc3': NC3(model, subset, num_classes),
        'nc4': NC4(model, subset, num_classes),
        'entropies': get_entropy(model, subset).tolist()}
    return metrics
    
    

