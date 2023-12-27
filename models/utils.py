import torch

class LinearizedEmbeddings(torch.nn.Module):
    def __init__(self, bert_embeddings, trainable):
        super().__init__()
        self.shapes = {1: bert_embeddings.word_embeddings.weight.data.size(),
                2: bert_embeddings.position_embeddings.weight.data.size(),
                3: bert_embeddings.token_type_embeddings.weight.data.size()}
        self.input_length = sum(shape[0] for shape in self.shapes.values())
        bert_word_embeddings = bert_embeddings.word_embeddings.weight.data.detach().clone().t()
        self.word_embeddings = torch.nn.Linear(self.shapes[1][0],
                self.shapes[1][1], bias=False)
        self.word_embeddings.weight.data = bert_word_embeddings.requires_grad_(trainable)
        bert_position_embeddings = bert_embeddings.position_embeddings.weight.data.detach().clone().t()
        self.position_embeddings = torch.nn.Linear(self.shapes[2][0],
                self.shapes[2][1], bias=False)
        self.position_embeddings.weight.data = bert_position_embeddings.requires_grad_(trainable)
        bert_token_type_embeddings = bert_embeddings.token_type_embeddings.weight.data.detach().clone().t()
        self.token_type_embeddings = torch.nn.Linear(self.shapes[3][0],
                self.shapes[3][1], bias=False)
        self.token_type_embeddings.weight.data = bert_token_type_embeddings.requires_grad_(trainable)
        
    def forward(self, X):
        X = torch.squeeze(X)
        X1 = X[:, sum(self.shapes[i][0] for i in range(1,1)):sum(self.shapes[i][0] for i in range(1,2))]
        X2 = X[:, sum(self.shapes[i][0] for i in range(1,2)):sum(self.shapes[i][0] for i in range(1,3))]
        X3 = X[:, sum(self.shapes[i][0] for i in range(1,3)):sum(self.shapes[i][0] for i in range(1,4))]
        word_out = self.word_embeddings(X1)
        position_out = self.position_embeddings(X2)
        token_type_out = self.token_type_embeddings(X3)
        return word_out + position_out + token_type_out


def stabilized_forward(linearized_encoder,
        min_=1e-1,
        max_=1e5,
        scaler=1e3,
        max_attempts=10):
    def forward(x):
        for i in range(len(linearized_encoder.layer)):
            attempts = 0
            prelim_x = linearized_encoder.layer[i](x)[0]
            while torch.mean(torch.abs(prelim_x))>max_ and attempts<max_attempts:
                for m in linearized_encoder.layer[i].modules():
                    if isinstance(m, torch.nn.Linear):
                        m.weight.data = m.weight.data/scaler
                prelim_x = linearized_encoder.layer[i](x)[0]
                attempts+=1
            while torch.mean(torch.abs(prelim_x))<min_ and attempts<max_attempts:
                for m in linearized_encoder.layer[i].modules():
                    if isinstance(m, torch.nn.Linear):
                        m.weight.data = scaler*m.weight.data
                prelim_x = linearized_encoder.layer[i](x)[0]
                attempts+=1
            assert attempts<max_attempts
            x = prelim_x
        return x
    return forward


def get_trainable_parameters(model):
    """ Returns a detached flattened tensor
        containing all trainable parameters.
    """
    parameters = []
    for module in model.modules():
        attr = []
        if hasattr(module, 'registered_parameters_name'):
            for name in module.registered_parameters_name:
                param = getattr(module, name)
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                attr.append(param.detach().reshape(-1))
        else:
            for _,param in module._parameters.items():
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                attr.append(param.detach().reshape(-1))
        if len(attr)<1:
            continue
        attr = torch.cat(attr)
        parameters.append(attr)
    parameters = torch.cat(parameters).reshape(1,-1)
    return parameters


def set_trainable_parameters(model, theta):
    """ Warning: this method sets trainable parameters only!
        Note that batchnorm statistics are not trainable.
        Not valid for model copying, use state_dict instead.
        
        Warning: When the model is flipped, this method sets
        potentially non-leaf tensors for model parameters.
        Thus, training / retriving gradients of the model
        parameters may be hindered.
        
        Warning: When the model is not flipped, any prior
        modifications made to theta are untracked because
        parameters of the model are updated in an untrackable
        fashion as param.data = theta[...]. 
    """
    theta = theta.reshape(1,-1)
    if not theta.requires_grad:
        theta = theta.requires_grad_(True)
    count = 0  
    for module in model.modules():
        if hasattr(module, 'registered_parameters_name'):
            for name in module.registered_parameters_name:
                param = getattr(module, name)
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                a = count
                b = a + param.numel()
                t = torch.reshape(theta[0,a:b], param.shape)
                setattr(module, name, t)
                count += param.numel()
        else:
            for named_param in module._parameters.items():
                name, param = named_param
                if param is None:
                    continue
                if not param.requires_grad:
                    continue
                a = count
                b = a + param.numel()
                t = torch.reshape(theta[0,a:b], param.shape)
                module._parameters[name].data = t
                count += param.numel()


def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded=(attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float())
    token_embeddings[input_mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, -2).values


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded=(attention_mask.unsqueeze(-1).float())
    sum_embeddings=torch.sum(token_embeddings * input_mask_expanded, -2)
    sum_mask=torch.clamp(input_mask_expanded.sum(-2), min=1e-9)
    return sum_embeddings / sum_mask


def pool(x, encoded_input, pool_type):
    if pool_type in ['avg', 'mean']:
        x = mean_pooling(x['last_hidden_state'], encoded_input['attention_mask'])
    elif pool_type == 'max':
        x = max_pooling(x['last_hidden_state'], encoded_input['attention_mask'])
    elif pool_type in ['cls', 'first']:
        x = x['last_hidden_state'][:,0]
    elif pool_type == 'pooler_output':
        x = x['pooler_output']
    else:
        raise ValueError(f'unknown pool type {pool_type}.')
    return x