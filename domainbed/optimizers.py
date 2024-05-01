import torch


def get_optimizer(name, params, **kwargs):
    name = name.lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    optim_cls = optimizers[name]
    if name=="sgd":
        return optim_cls(params, lr=1e-5,momentum=0.9)
    return optim_cls(params, **kwargs)
