from ..pruners.synflow import SynFlow
from .model_base import ClassificationModelBase
from .model_base import LinearizedClassificationModelBase
from .attention.linear import BertSelfAttentionLinear
from .utils import LinearizedEmbeddings
from .utils import pool, stabilized_forward
from .utils import get_trainable_parameters, set_trainable_parameters
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig
from copy import deepcopy
import torch.nn.functional as F
import torch
import math


class ClassifierBERT(ClassificationModelBase):
    def __init__(self, backbone_name,
                    pool_type,
                    num_classes,
                    freeze_embeddings,
                    prune_classifier,
                    attention_class,
                    device,
                    **kwargs):
        super().__init__()
        self.pool_type = pool_type
        bert = BertModel.from_pretrained(backbone_name)
        self.config = bert.config
        for layer_id in range(self.config.num_hidden_layers):
            w = get_trainable_parameters(bert.encoder.layer[layer_id].attention.self)
            bert.encoder.layer[layer_id].attention.self = attention_class(self.config)
            set_trainable_parameters(bert.encoder.layer[layer_id].attention.self, w)
        classifier = torch.nn.Linear(self.config.hidden_size, num_classes, bias=True)
        torch.nn.init.xavier_normal_(classifier.weight.data,gain=math.sqrt(2))
        self.module_dict = torch.nn.ModuleDict(
                {'embeddings': bert.embeddings,
                 'encoder' :bert.encoder,
                 'classifier': classifier})
        self.device = device
        self.to(self.device)
        self.prunable_module_keys = list(self.module_dict.keys())
        self.freeze_embeddings = freeze_embeddings
        if freeze_embeddings:
            self.prunable_module_keys.remove('embeddings')
            for param in self.module_dict['embeddings'].parameters():
                param.requires_grad=False
        if not prune_classifier:
            self.prunable_module_keys.remove('classifier')
        self._create_masks()
    
    @property
    def effective_masks(self):
        self.apply_masks()
        pruner = SynFlow(num_iters=1)
        scores = pruner.get_scores(self)
        masks = [(score>0).float() for score in scores]
        return masks
    
    def get_config(self):
        return self.config
        
    def get_prunable_modules(self):
        prunable_modules = []
        for key in self.prunable_module_keys:
            for m in self.module_dict[key].modules():
                if isinstance(m, torch.nn.Embedding):
                    prunable_modules.append(m)
                if isinstance(m, torch.nn.Linear):
                    prunable_modules.append(m)
        return prunable_modules
        
    def _create_masks(self):
        self.masks = []
        for module in self.get_prunable_modules():
            size = module.weight.data.size()
            mask = torch.ones(size).to(self.device)
            self.masks.append(mask)

    def _check_compatibility(self, masks):
        assert len(self.masks) == len(masks), "attempt to update masks with incorrect length"
        for old_mask,new_mask in zip(self.masks, masks):
            new_mask = new_mask.to(old_mask.device)
            assert torch.sum(new_mask*(1-old_mask)) == 0, "new masks are incompatible"
    
    def update_masks(self, masks):
        self._check_compatibility(masks)
        for i,mask in enumerate(masks):
            self.masks[i] = mask.to(self.device)

    def apply_masks(self):
        for i,module in enumerate(self.get_prunable_modules()):
            masked_weight = torch.mul(module.weight.data, self.masks[i])
            module.weight.data = masked_weight
    
    def forward(self, inp, embeddings=False):
        self.apply_masks()
        attention_mask = inp['attention_mask'][:, None, None, :]
        x = self.module_dict['embeddings'](inp['input_ids'])
        x = self.module_dict['encoder'](x, attention_mask=attention_mask)
        x = pool(x, inp, self.pool_type)
        if embeddings:
            return x
        x = self.module_dict['classifier'](x)
        return x
    
    def predict(self, dataset, batch_size=8, embeddings=False):
        self.apply_masks()
        outputs = []
        dataloader = DataLoader(dataset, batch_size=batch_size)
        for i,(X,y) in enumerate(dataloader):
            output = self(X, embeddings=embeddings)
            outputs.append(output)
        outputs = torch.vstack(outputs)
        return outputs

    def gradients(self, dataset, create_graph=False):
        self.apply_masks()
        dataloader = DataLoader(dataset, batch_size=8)
        gradients = []
        for i,(X,y) in enumerate(dataloader):
            output = self(X)
            loss = F.cross_entropy(output, y)
            loss.backward(create_graph=create_graph)
            for j,module in enumerate(self.get_prunable_modules()):
                curr_layer_grad = module.weight.grad
                if i==0:
                    gradients.append(curr_layer_grad)
                else:
                    sum_ = gradients[j]*i
                    avg_ = (sum_+curr_layer_grad)/(i+1)
                    gradients[j] = avg_
                module.weight.grad = None
        return gradients

    def linearize(self):
        self.apply_masks()
        return LinearizedBERTClassifier(self,
                stabilize=True,
                freeze_embeddings=self.freeze_embeddings)

    
class LinearizedBERTClassifier(LinearizedClassificationModelBase):
    def __init__(self, reference_model, stabilize=False, freeze_embeddings=False):
        super().__init__()
        self.device = reference_model.device
        self.freeze_embeddings = freeze_embeddings
        self.stabilize = stabilize
        linearized_encoder = self._build_encoder(reference_model)
        linearized_embeddings = LinearizedEmbeddings(
                reference_model.module_dict['embeddings'],
                trainable=not self.freeze_embeddings)
        self.input_length = linearized_embeddings.input_length
        self.module_dict = torch.nn.ModuleDict(
                {'embeddings': linearized_embeddings,
                'encoder': linearized_encoder,
                'classifier': deepcopy(reference_model.module_dict['classifier'])})
        self.prunable_module_keys = reference_model.prunable_module_keys
        for m in self.modules():
            if hasattr(m, 'weight'):
                m.weight.data = torch.abs(m.weight.data)
                m.weight.grad = None
        self.to(self.device)
                
    def _build_encoder(self, reference_model):
        config = BertConfig(hidden_act=lambda x: x,
                hidden_dropout_prob=0,
                attention_probs_dropout_prob=0)
        linearized_encoder = BertModel(config).encoder
        for layer in linearized_encoder.layer:
            layer.attention.self = BertSelfAttentionLinear(config)
        if self.stabilize:
            linearized_encoder.forward = stabilized_forward(linearized_encoder)
        identity = torch.nn.Identity()
        weights = []
        for m in reference_model.module_dict['encoder'].modules():
            if isinstance(m, torch.nn.Linear):
                weights.append(m.weight.data.detach().clone())
        idx = 0
        for m in linearized_encoder.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data = weights[idx].requires_grad_(True)
                idx+=1
            if isinstance(m, torch.nn.LayerNorm):
                m.forward = identity.forward
        return linearized_encoder
    
    def get_prunable_modules(self):
        prunable_modules = []
        for key in self.prunable_module_keys:
            for m in self.module_dict[key].modules():
                if isinstance(m, torch.nn.Embedding):
                    prunable_modules.append(m)
                if isinstance(m, torch.nn.Linear):
                    prunable_modules.append(m)
        return prunable_modules
        
    def forward(self, inp):
        x = self.module_dict['embeddings'](inp)[None, :, :]
        x = self.module_dict['encoder'](x)
        mask = {'attention_mask': torch.ones(512).to(self.device)}
        if self.stabilize:
            x = {'last_hidden_state': x}
        x = pool(x, mask, 'mean')
        x = self.module_dict['classifier'](x)
        return x