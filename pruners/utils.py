import numpy as np
import torch


def random_pruning(sparsities, shapes, pruning_type='direct'):
    masks = []
    for shape, sparsity in zip(shapes, sparsities):
        count = int(torch.prod(shape))
        mask = torch.ones(count)
        idx_to_prune = np.random.choice(range(count),
                size=int(sparsity*count),
                replace=False)
        mask[idx_to_prune] = 0.
        shape = tuple(int(dim) for dim in shape)
        mask = mask.reshape(shape)
        masks.append(mask)
    return masks


def score_pruning(target_sparsity, scores, pruning_type='direct'):
    scores_flatten = np.concatenate([score.cpu().reshape(-1) for score in scores])
    threshold = np.quantile(scores_flatten, target_sparsity)
    masks = [(score>threshold).float() for score in scores]
    return masks


def effective_masks_dense(masks):
    for i,mask in enumerate(masks):
        assert len(mask.size())==2, f"found mask of shape {mask.size()}"
        masks[i] = mask.T
    units=[mask.shape[-2] for mask in masks]+[masks[-1].shape[-1]]
    next_layer=torch.ones((units[-1],))
    way_out=[next_layer]
    for mask in masks[::-1]:
        curr_mask=torch.matmul(mask,next_layer.view(len(next_layer),1))
        next_layer=torch.sum(curr_mask,dim=1)>0
        way_out.append(next_layer)
    way_out=way_out[::-1]
    prev_layer=torch.ones((units[0],))
    way_in=[prev_layer]
    for mask in masks:
        curr_mask=torch.matmul(prev_layer.view(1,len(prev_layer)),mask)
        prev_layer=torch.sum(curr_mask,dim=0)>0
        way_in.append(prev_layer)
    activity=[w_in*w_out for w_in,w_out in zip(way_in,way_out)]
    effective_masks = []
    for i,mask in enumerate(masks):
        activity_prev = activity[i].view(len(activity[i]),1)
        activity_next = activity[i+1].view(1,len(activity[i+1]))
        effective_mask = mask*torch.matmul(activity_prev, activity_next)
        effective_masks.append(effective_mask.T)
    return effective_masks