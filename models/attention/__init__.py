from .linear import BertSelfAttentionLinear
from .original import BertSelfAttentionOriginal
from .masked_std import BertSelfAttentionMaskedStd
from .simple_std import BertSelfAttentionSimpleStd
from .huggingface import BertSelfAttentionHuggingface

def AttentionFactory(attention_name, **kwargs):
    implemented_attention = {}
    for _class_ in [
            BertSelfAttentionLinear,
            BertSelfAttentionOriginal,
            BertSelfAttentionMaskedStd,
            BertSelfAttentionSimpleStd,
            BertSelfAttentionHuggingface]:
        implemented_attention[_class_.__name__] = _class_
    if attention_name in implemented_attention:
        return implemented_attention[attention_name]
    else:
        raise NotImplementedError(f"attention {attention_name} is unknown")