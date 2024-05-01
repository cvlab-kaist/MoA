# Copyright (c) Kakao Brain. All Rights Reserved.

import timm
import torch
import torch.nn as nn
import torchvision.models


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def torchhub_load(repo, model, **kwargs):
    try:
        # torch >= 1.10
        network = torch.hub.load(repo, model=model, skip_validation=True, **kwargs)
    except TypeError:
        # torch 1.7.1
        network = torch.hub.load(repo, model=model, **kwargs)

    return network


def get_backbone(name, preserve_readout, pretrained, r=4):
    # print(r)
    if not pretrained:
        assert name in ["resnet50", "swag_regnety_16gf"], "Only RN50/RegNet supports non-pretrained network"

    if name == "resnet18":
        network = torchvision.models.resnet18(pretrained=True)
        n_outputs = 512
    elif name == "resnet50":
        network = torchvision.models.resnet50(pretrained=pretrained)
        n_outputs = 2048
    elif name == "resnet50_barlowtwins":
        network = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        n_outputs = 2048
    elif name == "resnet50_moco":
        network = torchvision.models.resnet50()

        # download pretrained model of MoCo v3: https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar
        ckpt_path = "./r-50-1000ep.pth.tar"

        # https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py#L172
        print("=> loading checkpoint '{}'".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        linear_keyword = "fc"  # resnet linear keyword
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = network.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(ckpt_path))

        n_outputs = 2048

    elif name == "nf_vitbase":
        network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True, pretrained_strict=False)
        n_outputs = network.num_features
        network.head = Identity()

    elif name == "nf_vitbase_lora":
        from domainbed.models.lora.lora_util import inject_trainable_lora

        network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        n_outputs = network.num_features
        network.head = Identity()
        for name, param in network.named_parameters():
            param.requires_grad = False

        require_grad_params, names = inject_trainable_lora(network, r=r)
        setattr(network, 'require_grad_params', require_grad_params)

    elif name == "nf_vitbase_moelora_last_qkv":
        from domainbed.models.lora.lora_util import inject_trainable_moe_lora

        network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        n_outputs = network.num_features
        network.head = Identity()
        for name, param in network.named_parameters():
            param.requires_grad = False
        require_grad_params, names = inject_trainable_moe_lora(network, r=[1, 2, 4, 8], where='last_qkv')
        setattr(network, 'require_grad_params', require_grad_params)

    elif name == "nf_vitbase_moek_every_qkv_new":
        from domainbed.models.lora.lora_util import inject_trainable_moe_kronecker_new

        network = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)
        n_outputs = network.num_features
        network.head = Identity()
        for name, param in network.named_parameters():
            param.requires_grad = False
        require_grad_params, names = inject_trainable_moe_kronecker_new(network, r=[1, 2, 4, 8], where='every_qkv')
        setattr(network, 'require_grad_params', require_grad_params)

    else:
        raise ValueError(name)

    if not preserve_readout:
        # remove readout layer (but left GAP and flatten)
        # final output shape: [B, n_outputs]
        if name.startswith("resnet"):
            del network.fc
            network.fc = Identity()

    return network, n_outputs
