# Code from https://github.com/cloneofsimo/LoRA
import json
import math
import random
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions.normal import Normal

# from tutel import moe as tutel_moe
# from tutel import net as tutel_net

try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save

    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )

    safetensors_available = False


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


def kronecker_product_einsum_batched(A: torch.Tensor, B: torch.Tensor):
    """
    Batched Version of Kronecker Products
    :param A: has shape (b, a, c)
    :param B: has shape (b, k, p)
    :return: (b, ak, cp)
    """
    assert A.dim() == 3 and B.dim() == 3
    res = torch.einsum('bac,bkp->bakcp', A, B).view(A.size(0), A.size(1) * B.size(1), A.size(2) * B.size(2))
    return res


def glorot_uniform(tensor: torch.Tensor):
    return torch.nn.init.xavier_uniform_(tensor, gain=math.sqrt(2))


class CosineTopKGate(torch.nn.Module):

    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, proj_dim=256, init_t=0.5, **options):
        super(CosineTopKGate, self).__init__()
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1.0 / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            cosine_projector = self.cosine_projector.float()
            sim_matrix = self.sim_matrix.float()
        else:
            cosine_projector = self.cosine_projector
            sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1), F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits


class K_Linear_MoE_new_(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 1,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        phm_dim = 64
        self.phm_dim = phm_dim
        self._in_feats_per_axis = in_features // phm_dim
        self._out_feats_per_axis = (out_features) // phm_dim
        self.phm_rank = 1
        self.n = 4
        self.proj_adapter1_left = []
        self.proj_adapter1_right = []
        self.kdropout = []
        self.router = CosineTopKGate(in_features, self.n)
        if self.r > 0:
            for i in range(self.n):
                self.proj_adapter1_left.append(
                    nn.Parameter(
                        torch.Tensor(size=(phm_dim, self._in_feats_per_axis, self.phm_rank * (2**i))),
                        requires_grad=True,
                    )
                )
                self.proj_adapter1_right.append(
                    nn.Parameter(
                        torch.Tensor(size=(phm_dim, self.phm_rank * (2**i), self._out_feats_per_axis)),
                        requires_grad=True,
                    )
                )

        self.kdropout = nn.Dropout(0.5)
        self.proj_adapter1_left = nn.ParameterList(self.proj_adapter1_left)
        self.proj_adapter1_right = nn.ParameterList(self.proj_adapter1_right)

        self.scaling = 1
        self.weight.requires_grad = False
        self.b_adapter = nn.ParameterList([nn.Parameter(torch.Tensor(out_features))])
        self.reset_parameters()
        self.init_W()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        # self.init_W()

    def init_W(self):
        for j in range(self.n):
            for i in range(self.phm_dim):
                self.proj_adapter1_left[j].data[i] = glorot_uniform(self.proj_adapter1_left[j].data[i])
                self.proj_adapter1_right[j].data[i] = glorot_uniform(self.proj_adapter1_right[j].data[i])
        self.b_adapter[0].data = torch.zeros_like(self.b_adapter[0].data)

    def set_phm_rule(self, phm_rule0_right=None, phm_rule0_left=None):
        self.phm_rule0_right = phm_rule0_right
        self.phm_rule0_left = phm_rule0_left

    def set_H(self, i, zero_pad=False):
        W = torch.bmm(self.proj_adapter1_left[i], self.proj_adapter1_right[i])

        phm_rule0 = torch.bmm(self.phm_rule1_left, self.phm_rule1_right)

        H = kronecker_product_einsum_batched(phm_rule0, W).sum(0)

        return self.kdropout(H)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            input_reshaped = rearrange(x, 'b c d -> (b c) d')
            logits = self.router(input_reshaped)
            if self.training and 1.0 > 0:
                logits_w = logits + 1.0 * torch.randn_like(logits) / self.n
            else:
                logits_w = logits
            logits_w = F.softmax(logits_w, dim=1)
            max_logits, top1_indices = torch.max(logits_w, 1, True)
            ret = torch.zeros_like(logits_w).scatter_(1, top1_indices, 1.0)
            ret = ret * logits_w
            ret = rearrange(ret, '(b c) e -> b c e', b=x.shape[0])
            l = []

            for i in range(self.n):
                l.append(torch.matmul(input=x, other=self.set_H(i)).unsqueeze(2))
            l = torch.concat(l, dim=2)
            res = ret.unsqueeze(-1) * l
            res = res.sum(dim=2)
            self.aux_loss = 0.01 * load_importance_loss(F.softmax(logits, dim=1), max_logits, 4, 1.0)

            return result + res + self.b_adapter[0]


def load_importance_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
    def load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise):
        assert gate_noise > 0, "`gate_noise` must be > 0 for normalization in load_importance_loss()."
        normal = Normal(
            torch.tensor([0.0], device=scores_wo_noise.device),
            torch.tensor([gate_noise / num_global_experts], device=scores_wo_noise.device),
        )
        threshold = topk_logits[:, -1].view(-1, 1).float()
        diff = scores_wo_noise.float() - threshold.float()
        prob = normal.cdf(diff)
        Load = prob.sum(0)
        l_load = Load.float().var() / (Load.float().mean() ** 2 + 1e-10)
        return l_load

    def importance_loss(scores_wo_noise):
        Impi = scores_wo_noise.float().sum(0)
        l_imp = Impi.float().var() / (Impi.float().mean() ** 2 + 1e-10)

        return l_imp

    l_imp = importance_loss(scores_wo_noise)
    l_load = load_loss(scores_wo_noise, topk_logits, num_global_experts, gate_noise)
    return (l_imp + l_load) / 2.0


class LoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, lora_bias=False):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}")

        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=lora_bias)
        self.lora_up = nn.Linear(r, out_features, bias=lora_bias)
        self.scale = 1.0

        nn.init.normal_(self.lora_down.weight, std=1 / r)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, input):
        return self.linear(input) + self.lora_up(self.lora_down(input)) * self.scale


class MultiLoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=[1, 2, 4, 8], lora_bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        lora_down_list = []
        lora_up_list = []
        for i in r:
            lora_down_list.append(nn.Linear(in_features, i, bias=lora_bias))
            lora_up_list.append(nn.Linear(i, out_features, bias=lora_bias))
            nn.init.normal_(lora_down_list[-1].weight, std=1 / i)
            nn.init.zeros_(lora_up_list[-1].weight)
        self.lora_down = nn.ModuleList(lora_down_list)
        self.lora_up = nn.ModuleList(lora_up_list)
        self.r_len = len(r)
        self.r = r
        self.scale = nn.Parameter(torch.zeros(4))
        self.sm = nn.Softmax(dim=0)

    def forward(self, input):
        x = self.linear(input)
        scale = self.sm(self.scale)
        for i in range(self.r_len):
            x = x + self.lora_up[i](self.lora_down[i](input)) * scale[i]
        return x


class MOELoraInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=[1, 2, 4, 8], lora_bias=False):
        super().__init__()

        self.router = CosineTopKGate(in_features, len(r))
        self.linear = nn.Linear(in_features, out_features, bias)
        lora_down_list = []
        lora_up_list = []
        for i in r:
            lora_down_list.append(nn.Linear(in_features, i, bias=lora_bias))
            lora_up_list.append(nn.Linear(i, out_features, bias=lora_bias))
            nn.init.normal_(lora_down_list[-1].weight, std=1 / i)
            nn.init.zeros_(lora_up_list[-1].weight)
        self.lora_down = nn.ModuleList(lora_down_list)
        self.lora_up = nn.ModuleList(lora_up_list)
        self.r_len = len(r)
        self.r = r

    def forward(self, input):
        x = self.linear(input)
        input_reshaped = rearrange(input, 'b c d -> (b c) d')
        logits = self.router(input_reshaped)
        if self.training and 1.0 > 0:
            logits_w = logits + 1.0 * torch.randn_like(logits) / self.r_len
        else:
            logits_w = logits
        logits_w = F.softmax(logits_w, dim=1)
        max_logits, top1_indices = torch.max(logits_w, 1, True)
        ret = torch.zeros_like(logits_w).scatter_(1, top1_indices, 1.0)
        ret = ret * logits_w
        ret = rearrange(ret, '(b c) e -> b c e', b=input.shape[0])
        l = []
        for i in range(self.r_len):
            l.append(self.lora_up[i](self.lora_down[i](input)).unsqueeze(2))
        l = torch.concat(l, dim=2)
        res = ret.unsqueeze(-1) * l
        res = res.sum(dim=2)
        x = x + res
        self.aux_loss = 0.01 * load_importance_loss(F.softmax(logits, dim=1), max_logits, 4, 1.0)
        return x


UNET_DEFAULT_TARGET_REPLACE = {"CrossAttention", "Attention", "GEGLU"}
TEXT_ENCODER_DEFAULT_TARGET_REPLACE = {"CLIPAttention"}

DEFAULT_TARGET_REPLACE = UNET_DEFAULT_TARGET_REPLACE

EMBED_FLAG = "<embed>"


def _find_children(
    model,
    search_class: List[Type[nn.Module]] = [nn.Linear],
):
    """
    Find all modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """
    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for parent in model.modules():
        for name, module in parent.named_children():
            if any([isinstance(module, _class) for _class in search_class]):
                yield parent, name, module


def _find_modules_v2(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    """
    Find all modules of a certain class (or union of classes) that are direct or
    indirect descendants of other modules of a certain class (or union of classes).

    Returns all matching modules, along with the parent of those moduless and the
    names they are referenced by.
    """

    # Get the targets we should replace all linears under
    ancestors = (module for module in model.modules() if module.__class__.__name__ in ancestor_class)

    # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
    for ancestor in ancestors:
        for fullname, module in ancestor.named_modules():
            if any([isinstance(module, _class) for _class in search_class]):
                # Find the direct parent if this is a descendant, not a child, of target
                *path, name = fullname.split(".")
                parent = ancestor
                while path:
                    parent = parent.get_submodule(path.pop(0))
                # Skip this linear if it's a child of a LoraInjectedLinear
                if exclude_children_of and any([isinstance(parent, _class) for _class in exclude_children_of]):
                    continue
                # Otherwise, yield it
                yield parent, name, module


def _find_modules_old(
    model,
    ancestor_class: Set[str] = DEFAULT_TARGET_REPLACE,
    search_class: List[Type[nn.Module]] = [nn.Linear],
    exclude_children_of: Optional[List[Type[nn.Module]]] = [LoraInjectedLinear],
):
    ret = []
    for _module in model.modules():
        if _module.__class__.__name__ in ancestor_class:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__ in search_class:
                    ret.append((_module, name, _child_module))
    # print(ret)
    return ret


_find_modules = _find_modules_v2


def inject_trainable_multi_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r=[1, 2, 4, 8],
    loras=None,  # path to lora .pt
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = MultiLoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r,
            lora_bias=False,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        for i in range(len(r)):
            if loras != None:
                _module._modules[name].lora_up.weight = loras.pop(0)
                _module._modules[name].lora_down.weight = loras.pop(0)
            _module._modules[name].lora_up[i].weight.requires_grad = True
            _module._modules[name].lora_down[i].weight.requires_grad = True
        names.append(name)
    return require_grad_params, names


def inject_trainable_moe_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r=[1, 2, 4, 8],
    loras=None,  # path to lora .pt
    where=None,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []
    ii = 0
    if loras != None:
        loras = torch.load(loras)
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        if not where:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = MOELoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r,
                lora_bias=False,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            require_grad_params.append(_module._modules[name].lora_up.parameters())
            require_grad_params.append(_module._modules[name].lora_down.parameters())
            require_grad_params.append(_module._modules[name].router.parameters())

            for i in range(len(r)):
                if loras != None:
                    _module._modules[name].lora_up.weight = loras.pop(0)
                    _module._modules[name].lora_down.weight = loras.pop(0)
                _module._modules[name].lora_up[i].weight.requires_grad = True
                _module._modules[name].lora_down[i].weight.requires_grad = True
            for params in _module._modules[name].router.parameters():
                params.requires_grad = True
            names.append(name)

        elif 'every' in where:
            if (ii % 4 == 0) or (ii % 4 == 1):
                if where == 'every':
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = MOELoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        lora_bias=False,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    require_grad_params.append(_module._modules[name].lora_up.parameters())
                    require_grad_params.append(_module._modules[name].lora_down.parameters())
                    require_grad_params.append(_module._modules[name].router.parameters())

                    for i in range(len(r)):
                        if loras != None:
                            _module._modules[name].lora_up.weight = loras.pop(0)
                            _module._modules[name].lora_down.weight = loras.pop(0)
                        _module._modules[name].lora_up[i].weight.requires_grad = True
                        _module._modules[name].lora_down[i].weight.requires_grad = True
                    for params in _module._modules[name].router.parameters():
                        params.requires_grad = True
                    names.append(name)

                elif name in where:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = MOELoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        lora_bias=False,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    require_grad_params.append(_module._modules[name].lora_up.parameters())
                    require_grad_params.append(_module._modules[name].lora_down.parameters())
                    require_grad_params.append(_module._modules[name].router.parameters())

                    for i in range(len(r)):
                        if loras != None:
                            _module._modules[name].lora_up.weight = loras.pop(0)
                            _module._modules[name].lora_down.weight = loras.pop(0)
                        _module._modules[name].lora_up[i].weight.requires_grad = True
                        _module._modules[name].lora_down[i].weight.requires_grad = True
                    for params in _module._modules[name].router.parameters():
                        params.requires_grad = True
                    names.append(name)

        elif 'last' in where:
            if 19 < ii:
                if where == 'last':
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = MOELoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        lora_bias=False,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    require_grad_params.append(_module._modules[name].lora_up.parameters())
                    require_grad_params.append(_module._modules[name].lora_down.parameters())
                    require_grad_params.append(_module._modules[name].router.parameters())

                    for i in range(len(r)):
                        if loras != None:
                            _module._modules[name].lora_up.weight = loras.pop(0)
                            _module._modules[name].lora_down.weight = loras.pop(0)
                        _module._modules[name].lora_up[i].weight.requires_grad = True
                        _module._modules[name].lora_down[i].weight.requires_grad = True
                    for params in _module._modules[name].router.parameters():
                        params.requires_grad = True
                    names.append(name)

                elif name in where:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = MOELoraInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        lora_bias=False,
                    )
                    _tmp.linear.weight = weight
                    if bias is not None:
                        _tmp.linear.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    require_grad_params.append(_module._modules[name].lora_up.parameters())
                    require_grad_params.append(_module._modules[name].lora_down.parameters())
                    require_grad_params.append(_module._modules[name].router.parameters())

                    for i in range(len(r)):
                        if loras != None:
                            _module._modules[name].lora_up.weight = loras.pop(0)
                            _module._modules[name].lora_down.weight = loras.pop(0)
                        _module._modules[name].lora_up[i].weight.requires_grad = True
                        _module._modules[name].lora_down[i].weight.requires_grad = True
                    for params in _module._modules[name].router.parameters():
                        params.requires_grad = True
                    names.append(name)

        elif name == where:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = MOELoraInjectedLinear(
                _child_module.in_features,
                _child_module.out_features,
                _child_module.bias is not None,
                r,
                lora_bias=False,
            )
            _tmp.linear.weight = weight
            if bias is not None:
                _tmp.linear.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            require_grad_params.append(_module._modules[name].lora_up.parameters())
            require_grad_params.append(_module._modules[name].lora_down.parameters())
            require_grad_params.append(_module._modules[name].router.parameters())

            for i in range(len(r)):
                if loras != None:
                    _module._modules[name].lora_up.weight = loras.pop(0)
                    _module._modules[name].lora_down.weight = loras.pop(0)
                _module._modules[name].lora_up[i].weight.requires_grad = True
                _module._modules[name].lora_down[i].weight.requires_grad = True
            for params in _module._modules[name].router.parameters():
                params.requires_grad = True
            names.append(name)
        ii += 1

    return require_grad_params, names


def set_phm_rule(phm_dim=64):
    phm_rule0_left = []
    phm_rule0_right = []
    phm_rule1_left = []
    phm_rule1_right = []
    phm_rule2_left = []
    phm_rule2_right = []
    phm_rule3_left = []
    phm_rule3_right = []
    for i in range(4):
        phm_rule0_left.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), phm_dim // (2**i), 1), requires_grad=True)
        )
        phm_rule0_right.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), 1, phm_dim // (2**i)), requires_grad=True)
        )
        phm_rule0_left[-1].data.uniform_(-0.01, 0.01)
        phm_rule0_right[-1].data.uniform_(-0.01, 0.01)

        phm_rule1_left.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), phm_dim // (2**i), 1), requires_grad=True)
        )
        phm_rule1_right.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), 1, phm_dim // (2**i)), requires_grad=True)
        )
        phm_rule1_left[-1].data.uniform_(-0.01, 0.01)
        phm_rule1_right[-1].data.uniform_(-0.01, 0.01)

        phm_rule2_left.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), phm_dim // (2**i), 1), requires_grad=True)
        )
        phm_rule2_right.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), 1, phm_dim // (2**i)), requires_grad=True)
        )
        phm_rule2_left[-1].data.uniform_(-0.01, 0.01)
        phm_rule2_right[-1].data.uniform_(-0.01, 0.01)

        phm_rule3_left.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), phm_dim // (2**i), 1), requires_grad=True)
        )
        phm_rule3_right.append(
            nn.Parameter(torch.FloatTensor(phm_dim // (2**i), 1, phm_dim // (2**i)), requires_grad=True)
        )
        phm_rule3_left[-1].data.uniform_(-0.01, 0.01)
        phm_rule3_right[-1].data.uniform_(-0.01, 0.01)

    return {
        'phm_rule0_left': nn.ParameterList(phm_rule0_left),
        'phm_rule0_right': nn.ParameterList(phm_rule0_right),
        'phm_rule1_left': nn.ParameterList(phm_rule1_left),
        'phm_rule1_right': nn.ParameterList(phm_rule1_right),
        'phm_rule2_left': nn.ParameterList(phm_rule2_left),
        'phm_rule2_right': nn.ParameterList(phm_rule2_right),
        'phm_rule3_left': nn.ParameterList(phm_rule3_left),
        'phm_rule3_right': nn.ParameterList(phm_rule3_right),
    }


def set_phm_rule_new(phm_dim=64):
    phm_rule0_left = []
    phm_rule0_right = []
    phm_rule1_left = []
    phm_rule1_right = []

    phm_rule0_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1), requires_grad=True)
    phm_rule0_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim), requires_grad=True)
    phm_rule0_left.data.uniform_(-0.01, 0.01)
    phm_rule0_right.data.uniform_(-0.01, 0.01)

    phm_rule1_left = nn.Parameter(torch.FloatTensor(phm_dim, phm_dim, 1), requires_grad=True)
    phm_rule1_right = nn.Parameter(torch.FloatTensor(phm_dim, 1, phm_dim), requires_grad=True)
    phm_rule1_left.data.uniform_(-0.01, 0.01)
    phm_rule1_right.data.uniform_(-0.01, 0.01)
    return {
        'phm_rule0_left': phm_rule0_left,
        'phm_rule0_right': phm_rule0_right,
        'phm_rule1_left': phm_rule1_left,
        'phm_rule1_right': phm_rule1_right,
    }


def inject_trainable_moe_kronecker_new(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r=[1, 2, 4, 8],
    loras=None,  # path to lora .pt
    where=None,
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []
    ii = 0
    k = set_phm_rule_new()
    for key, value in k.items():
        require_grad_params.append(value)
        setattr(model, key, value)

    if loras != None:
        loras = torch.load(loras)
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        if name == 'qkv':
            layer = K_Linear_MoE_new_

        if not where:  # ALL
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = layer(
                _child_module.in_features,
                _child_module.out_features,
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            require_grad_params.append(_module._modules[name].router.parameters())
            require_grad_params.append(_module._modules[name].b_adapter.parameters())

            names.append(name)
            for key, value in k.items():
                setattr(_module._modules[name], key, value)

        elif 'every' in where:  # every 2
            if (ii % 4 == 0) or (ii % 4 == 1):
                if where == 'every':
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = layer(
                        _child_module.in_features,
                        _child_module.out_features,
                    )
                    _tmp.weight = weight
                    if bias is not None:
                        _tmp.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    names.append(name)
                    for key, value in k.items():
                        setattr(_module._modules[name], key, value)

                elif name in where:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = layer(
                        _child_module.in_features,
                        _child_module.out_features,
                    )
                    _tmp.weight = weight
                    if bias is not None:
                        _tmp.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    names.append(name)
                    for key, value in k.items():
                        setattr(_module._modules[name], key, value)

        elif 'last' in where:
            if 19 < ii:
                if where == 'last':
                    # import pdb; pdb.set_trace()
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = layer(
                        _child_module.in_features,
                        _child_module.out_features,
                    )
                    _tmp.weight = weight
                    if bias is not None:
                        _tmp.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    names.append(name)
                    for key, value in k.items():
                        setattr(_module._modules[name], key, value)
                    # import pdb; pdb.set_trace()
                elif name in where:
                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = layer(
                        _child_module.in_features,
                        _child_module.out_features,
                    )
                    _tmp.weight = weight
                    if bias is not None:
                        _tmp.bias = bias

                    # switch the module
                    _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
                    _module._modules[name] = _tmp

                    require_grad_params.append(_module._modules[name].router.parameters())
                    require_grad_params.append(_module._modules[name].b_adapter.parameters())

                    names.append(name)
                    for key, value in k.items():
                        setattr(_module._modules[name], key, value)

        elif name == where:
            weight = _child_module.weight
            bias = _child_module.bias
            _tmp = layer(
                _child_module.in_features,
                _child_module.out_features,
            )
            _tmp.weight = weight
            if bias is not None:
                _tmp.bias = bias

            # switch the module
            _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
            _module._modules[name] = _tmp

            names.append(name)
            for key, value in k.items():
                setattr(_module._modules[name], key, value)
        ii += 1

    return require_grad_params, names


def inject_trainable_lora(
    model: nn.Module,
    target_replace_module: Set[str] = DEFAULT_TARGET_REPLACE,
    r: int = 4,
    loras=None,  # path to lora .pt
):
    """
    inject lora into model, and returns lora parameter groups.
    """

    require_grad_params = []
    names = []

    if loras != None:
        loras = torch.load(loras)
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r,
            lora_bias=False,
        )
        _tmp.linear.weight = weight
        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _tmp.to(_child_module.weight.device).to(_child_module.weight.dtype)
        _module._modules[name] = _tmp

        require_grad_params.append(_module._modules[name].lora_up.parameters())
        require_grad_params.append(_module._modules[name].lora_down.parameters())

        if loras != None:
            _module._modules[name].lora_up.weight = loras.pop(0)
            _module._modules[name].lora_down.weight = loras.pop(0)

        _module._modules[name].lora_up.weight.requires_grad = True
        _module._modules[name].lora_down.weight.requires_grad = True
        names.append(name)
    return require_grad_params, names


def extract_lora_ups_down(model, target_replace_module=DEFAULT_TARGET_REPLACE):

    loras = []

    for _m, _n, _child_module in _find_modules(model, target_replace_module, search_class=[LoraInjectedLinear]):
        loras.append((_child_module.lora_up, _child_module.lora_down))

    if len(loras) == 0:
        raise ValueError("No lora injected.")

    return loras


def save_lora_weight(
    model,
    path="./lora.pt",
    target_replace_module=DEFAULT_TARGET_REPLACE,
):
    weights = []
    for _up, _down in extract_lora_ups_down(model, target_replace_module=target_replace_module):
        weights.append(_up.weight.to("cpu").to(torch.float16))
        weights.append(_down.weight.to("cpu").to(torch.float16))

    torch.save(weights, path)


def save_lora_as_json(model, path="./lora.json"):
    weights = []
    for _up, _down in extract_lora_ups_down(model):
        weights.append(_up.weight.detach().cpu().numpy().tolist())
        weights.append(_down.weight.detach().cpu().numpy().tolist())

    import json

    with open(path, "w") as f:
        json.dump(weights, f)


def save_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    Saves the Lora from multiple modules in a single safetensor file.

    modelmap is a dictionary of {
        "module name": (module, target_replace_module)
    }
    """
    weights = {}
    metadata = {}

    for name, (model, target_replace_module) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        for i, (_up, _down) in enumerate(extract_lora_ups_down(model, target_replace_module)):
            metadata[f"{name}:{i}:rank"] = str(_down.out_features)
            weights[f"{name}:{i}:up"] = _up.weight
            weights[f"{name}:{i}:down"] = _down.weight

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def save_safeloras(
    modelmap: Dict[str, Tuple[nn.Module, Set[str]]] = {},
    outpath="./lora.safetensors",
):
    return save_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def convert_loras_to_safeloras_with_embeds(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    embeds: Dict[str, torch.Tensor] = {},
    outpath="./lora.safetensors",
):
    """
    Converts the Lora from multiple pytorch .pt files into a single safetensor file.

    modelmap is a dictionary of {
        "module name": (pytorch_model_path, target_replace_module, rank)
    }
    """

    weights = {}
    metadata = {}

    for name, (path, target_replace_module, r) in modelmap.items():
        metadata[name] = json.dumps(list(target_replace_module))

        lora = torch.load(path)
        for i, weight in enumerate(lora):
            is_up = i % 2 == 0
            i = i // 2

            if is_up:
                metadata[f"{name}:{i}:rank"] = str(r)
                weights[f"{name}:{i}:up"] = weight
            else:
                weights[f"{name}:{i}:down"] = weight

    for token, tensor in embeds.items():
        metadata[token] = EMBED_FLAG
        weights[token] = tensor

    print(f"Saving weights to {outpath}")
    safe_save(weights, outpath, metadata)


def convert_loras_to_safeloras(
    modelmap: Dict[str, Tuple[str, Set[str], int]] = {},
    outpath="./lora.safetensors",
):
    convert_loras_to_safeloras_with_embeds(modelmap=modelmap, outpath=outpath)


def parse_safeloras(
    safeloras,
) -> Dict[str, Tuple[List[nn.parameter.Parameter], List[int], List[str]]]:
    """
    Converts a loaded safetensor file that contains a set of module Loras
    into Parameters and other information

    Output is a dictionary of {
        "module name": (
            [list of weights],
            [list of ranks],
            target_replacement_modules
        )
    }
    """
    loras = {}
    metadata = safeloras.metadata()

    get_name = lambda k: k.split(":")[0]

    keys = list(safeloras.keys())
    keys.sort(key=get_name)

    for name, module_keys in groupby(keys, get_name):
        info = metadata.get(name)

        if not info:
            raise ValueError(f"Tensor {name} has no metadata - is this a Lora safetensor?")

        # Skip Textual Inversion embeds
        if info == EMBED_FLAG:
            continue

        # Handle Loras
        # Extract the targets
        target = json.loads(info)

        # Build the result lists - Python needs us to preallocate lists to insert into them
        module_keys = list(module_keys)
        ranks = [4] * (len(module_keys) // 2)
        weights = [None] * len(module_keys)

        for key in module_keys:
            # Split the model name and index out of the key
            _, idx, direction = key.split(":")
            idx = int(idx)

            # Add the rank
            ranks[idx] = int(metadata[f"{name}:{idx}:rank"])

            # Insert the weight into the list
            idx = idx * 2 + (1 if direction == "down" else 0)
            weights[idx] = nn.parameter.Parameter(safeloras.get_tensor(key))

        loras[name] = (weights, ranks, target)

    return loras


def parse_safeloras_embeds(
    safeloras,
) -> Dict[str, torch.Tensor]:
    """
    Converts a loaded safetensor file that contains Textual Inversion embeds into
    a dictionary of embed_token: Tensor
    """
    embeds = {}
    metadata = safeloras.metadata()

    for key in safeloras.keys():
        # Only handle Textual Inversion embeds
        meta = metadata.get(key)
        if not meta or meta != EMBED_FLAG:
            continue

        embeds[key] = safeloras.get_tensor(key)

    return embeds


def load_safeloras(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras)


def load_safeloras_embeds(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras_embeds(safeloras)


def load_safeloras_both(path, device="cpu"):
    safeloras = safe_open(path, framework="pt", device=device)
    return parse_safeloras(safeloras), parse_safeloras_embeds(safeloras)


def weight_apply_lora(model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, alpha=1.0):

    for _m, _n, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        weight = _child_module.weight

        up_weight = loras.pop(0).detach().to(weight.device)
        down_weight = loras.pop(0).detach().to(weight.device)

        # W <- W + U * D
        weight = weight + alpha * (up_weight @ down_weight).type(weight.dtype)
        _child_module.weight = nn.Parameter(weight)


def monkeypatch_lora(model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, r: int = 4):
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[nn.Linear]):
        weight = _child_module.weight
        bias = _child_module.bias
        _tmp = LoraInjectedLinear(
            _child_module.in_features,
            _child_module.out_features,
            _child_module.bias is not None,
            r=r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(up_weight.type(weight.dtype))
        _module._modules[name].lora_down.weight = nn.Parameter(down_weight.type(weight.dtype))

        _module._modules[name].to(weight.device)


def monkeypatch_replace_lora(model, loras, target_replace_module=DEFAULT_TARGET_REPLACE, r: int = 4):
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[LoraInjectedLinear]):
        weight = _child_module.linear.weight
        bias = _child_module.linear.bias
        _tmp = LoraInjectedLinear(
            _child_module.linear.in_features,
            _child_module.linear.out_features,
            _child_module.linear.bias is not None,
            r=r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(up_weight.type(weight.dtype))
        _module._modules[name].lora_down.weight = nn.Parameter(down_weight.type(weight.dtype))

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    r: Union[int, List[int]] = 4,
):
    for _module, name, _child_module in _find_modules(
        model, target_replace_module, search_class=[nn.Linear, LoraInjectedLinear]
    ):
        _source = _child_module.linear if isinstance(_child_module, LoraInjectedLinear) else _child_module

        weight = _source.weight
        bias = _source.bias
        _tmp = LoraInjectedLinear(
            _source.in_features,
            _source.out_features,
            _source.bias is not None,
            r=r.pop(0) if isinstance(r, list) else r,
        )
        _tmp.linear.weight = weight

        if bias is not None:
            _tmp.linear.bias = bias

        # switch the module
        _module._modules[name] = _tmp

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(up_weight.type(weight.dtype))
        _module._modules[name].lora_down.weight = nn.Parameter(down_weight.type(weight.dtype))

        _module._modules[name].to(weight.device)


def monkeypatch_or_replace_safeloras(models, safeloras):
    loras = parse_safeloras(safeloras)

    for name, (lora, ranks, target) in loras.items():
        model = getattr(models, name, None)

        if not model:
            print(f"No model provided for {name}, contained in Lora")
            continue

        monkeypatch_or_replace_lora(model, lora, target, ranks)


def monkeypatch_remove_lora(model):
    for _module, name, _child_module in _find_children(model, search_class=[LoraInjectedLinear]):
        _source = _child_module.linear
        weight, bias = _source.weight, _source.bias

        _tmp = nn.Linear(_source.in_features, _source.out_features, bias is not None)

        _tmp.weight = weight
        if bias is not None:
            _tmp.bias = bias

        _module._modules[name] = _tmp


def monkeypatch_add_lora(
    model,
    loras,
    target_replace_module=DEFAULT_TARGET_REPLACE,
    alpha: float = 1.0,
    beta: float = 1.0,
):
    for _module, name, _child_module in _find_modules(model, target_replace_module, search_class=[LoraInjectedLinear]):
        weight = _child_module.linear.weight

        up_weight = loras.pop(0)
        down_weight = loras.pop(0)

        _module._modules[name].lora_up.weight = nn.Parameter(
            up_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_up.weight.to(weight.device) * beta
        )
        _module._modules[name].lora_down.weight = nn.Parameter(
            down_weight.type(weight.dtype).to(weight.device) * alpha
            + _module._modules[name].lora_down.weight.to(weight.device) * beta
        )

        _module._modules[name].to(weight.device)


def tune_lora_scale(model, alpha: float = 1.0):
    for _module in model.modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            _module.scale = alpha


def _text_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["text_encoder", "pt"])


def _ti_lora_path(path: str) -> str:
    assert path.endswith(".pt"), "Only .pt files are supported"
    return ".".join(path.split(".")[:-1] + ["ti", "pt"])


def apply_learned_embed_in_clip(
    learned_embeds,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    if isinstance(token, str):
        trained_tokens = [token]
    elif isinstance(token, list):
        assert len(learned_embeds.keys()) == len(
            token
        ), "The number of tokens and the number of embeds should be the same"
        trained_tokens = token
    else:
        trained_tokens = list(learned_embeds.keys())

    for token in trained_tokens:
        print(token)
        embeds = learned_embeds[token]

        # cast to dtype of text_encoder
        dtype = text_encoder.get_input_embeddings().weight.dtype
        num_added_tokens = tokenizer.add_tokens(token)

        i = 1
        if not idempotent:
            while num_added_tokens == 0:
                print(f"The tokenizer already contains the token {token}.")
                token = f"{token[:-1]}-{i}>"
                print(f"Attempting to add the token {token}.")
                num_added_tokens = tokenizer.add_tokens(token)
                i += 1
        elif num_added_tokens == 0 and idempotent:
            print(f"The tokenizer already contains the token {token}.")
            print(f"Replacing {token} embedding.")

        # resize the token embeddings
        text_encoder.resize_token_embeddings(len(tokenizer))

        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


def load_learned_embed_in_clip(
    learned_embeds_path,
    text_encoder,
    tokenizer,
    token: Optional[Union[str, List[str]]] = None,
    idempotent=False,
):
    learned_embeds = torch.load(learned_embeds_path)
    apply_learned_embed_in_clip(learned_embeds, text_encoder, tokenizer, token, idempotent)


def patch_pipe(
    pipe,
    maybe_unet_path,
    token: Optional[str] = None,
    r: int = 4,
    patch_unet=True,
    patch_text=False,
    patch_ti=False,
    idempotent_token=True,
    unet_target_replace_module=DEFAULT_TARGET_REPLACE,
    text_target_replace_module=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
):
    if maybe_unet_path.endswith(".pt"):
        # torch format

        if maybe_unet_path.endswith(".ti.pt"):
            unet_path = maybe_unet_path[:-6] + ".pt"
        elif maybe_unet_path.endswith(".text_encoder.pt"):
            unet_path = maybe_unet_path[:-16] + ".pt"

        ti_path = _ti_lora_path(unet_path)
        text_path = _text_lora_path(unet_path)

        if patch_unet:
            print("LoRA : Patching Unet")
            monkeypatch_or_replace_lora(
                pipe.unet,
                torch.load(unet_path),
                r=r,
                target_replace_module=unet_target_replace_module,
            )

        if patch_text:
            print("LoRA : Patching text encoder")
            monkeypatch_or_replace_lora(
                pipe.text_encoder,
                torch.load(text_path),
                target_replace_module=text_target_replace_module,
                r=r,
            )
        if patch_ti:
            print("LoRA : Patching token input")
            token = load_learned_embed_in_clip(
                ti_path,
                pipe.text_encoder,
                pipe.tokenizer,
                token=token,
                idempotent=idempotent_token,
            )

    elif maybe_unet_path.endswith(".safetensors"):
        safeloras = safe_open(maybe_unet_path, framework="pt", device="cpu")
        monkeypatch_or_replace_safeloras(pipe, safeloras)
        tok_dict = parse_safeloras_embeds(safeloras)
        apply_learned_embed_in_clip(
            tok_dict,
            pipe.text_encoder,
            pipe.tokenizer,
            token=token,
            idempotent=idempotent_token,
        )


@torch.no_grad()
def inspect_lora(model):
    moved = {}

    for name, _module in model.named_modules():
        if _module.__class__.__name__ == "LoraInjectedLinear":
            ups = _module.lora_up.weight.data.clone()
            downs = _module.lora_down.weight.data.clone()

            wght: torch.Tensor = ups @ downs

            dist = wght.flatten().abs().mean().item()
            if name in moved:
                moved[name].append(dist)
            else:
                moved[name] = [dist]

    return moved


def save_all(
    unet,
    text_encoder,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    save_lora=True,
    save_ti=True,
    target_replace_module_text=TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
    target_replace_module_unet=DEFAULT_TARGET_REPLACE,
    safe_form=True,
):
    if not safe_form:
        # save ti
        if save_ti:
            ti_path = _ti_lora_path(save_path)
            learned_embeds_dict = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                learned_embeds_dict[tok] = learned_embeds.detach().cpu()

            torch.save(learned_embeds_dict, ti_path)
            print("Ti saved to ", ti_path)

        # save text encoder
        if save_lora:

            save_lora_weight(unet, save_path, target_replace_module=target_replace_module_unet)
            print("Unet saved to ", save_path)

            save_lora_weight(
                text_encoder,
                _text_lora_path(save_path),
                target_replace_module=target_replace_module_text,
            )
            print("Text Encoder saved to ", _text_lora_path(save_path))

    else:
        assert save_path.endswith(".safetensors"), f"Save path : {save_path} should end with .safetensors"

        loras = {}
        embeds = None

        if save_lora:

            loras["unet"] = (unet, target_replace_module_unet)
            loras["text_encoder"] = (text_encoder, target_replace_module_text)

        if save_ti:
            embeds = {}
            for tok, tok_id in zip(placeholder_tokens, placeholder_token_ids):
                learned_embeds = text_encoder.get_input_embeddings().weight[tok_id]
                print(
                    f"Current Learned Embeddings for {tok}:, id {tok_id} ",
                    learned_embeds[:4],
                )
                embeds[tok] = learned_embeds.detach().cpu()

        save_safeloras_with_embeds(loras, embeds, save_path)
