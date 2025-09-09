import torch
from torch import optim as optim
from muon import MuonWithAuxAdam

def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config['optimizer']['type'].lower()
    optimizer = None
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config['optimizer']['eps'], betas=config['optimizer']['betas'],
                                lr=config['base_lr'], weight_decay=config['weight_decay'])
    elif opt_lower == 'muon':
        hidden_weights = [p for p in model.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                lr=config['base_lr'], weight_decay=0.01),
            dict(params=hidden_gains_biases, use_muon=False,
                lr=config['base_lr'], betas=config['optimizer']['betas'], weight_decay=config['weight_decay']),
        ]
        optimizer = MuonWithAuxAdam(param_groups)
    else:
        raise NotImplementedError(f"Optimizer {opt_lower} not supported")

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin