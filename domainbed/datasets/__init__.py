import numpy as np
import torch

from domainbed.datasets import datasets
from domainbed.datasets import transforms as DBT
from domainbed.lib import misc


def set_transfroms(dset, data_type, hparams, algorithm_class=None,args=None):
    """
    Args:
        data_type: ['train', 'valid', 'test', 'mnist']
    """
    assert hparams["data_augmentation"]

    additional_data = False
    if data_type == "train":
        if "base" in hparams.model or "vit"==hparams.model  :
            dset.transforms = {"x": DBT.aug_open}
        elif "large" in hparams.model :
            dset.transforms = {"x": DBT.aug_img}
        additional_data = True
    elif data_type == "valid":
        if hparams["val_augment"] is False:
            if "base" in hparams.model or "vit"==hparams.model :
                dset.transforms = {"x": DBT.basic_open}
            elif "large" in hparams.model :
                dset.transforms = {"x": DBT.basic_img}
        else:
            # Originally, DomainBed use same training augmentation policy to validation.
            # We turn off the augmentation for validation as default,
            # but left the option to reproducibility.
            if "base" in hparams.model or "vit"==hparams.model  :
                dset.transforms = {"x": DBT.aug_open}
            elif "large" in hparams.model :
                dset.transforms = {"x": DBT.aug_img}
    elif data_type == "test":
        if "base" in hparams.model or "vit"==hparams.model :
            dset.transforms = {"x": DBT.basic_open}
        elif "large" in hparams.model :
            dset.transforms = {"x": DBT.basic_img}
    elif data_type == "mnist":
        # No augmentation for mnist
        dset.transforms = {"x": lambda x: x}
    else:
        raise ValueError(data_type)

    if additional_data and algorithm_class is not None:
        for key, transform in algorithm_class.transforms.items():
            dset.transforms[key] = transform


def get_dataset(test_envs, args, hparams, algorithm_class=None):
    """Get dataset and split."""
    is_mnist = "MNIST" in args.dataset
    dataset = vars(datasets)[args.dataset](args.data_dir)
    #  if not isinstance(dataset, MultipleEnvironmentImageFolder):
    #      raise ValueError("SMALL image datasets are not implemented (corrupted), for transform.")

    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):

        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
            args
        )
        if env_i in test_envs:
            in_type = "test"
            out_type = "test"
        else :
            in_type = "train"
            out_type = "valid"
        # else:
        #     continue
        
        if is_mnist:
            in_type = "mnist"
            out_type = "mnist"

        set_transfroms(in_, in_type, hparams, algorithm_class,args)
        set_transfroms(out, out_type, hparams, algorithm_class,args)

        if hparams["class_balanced"]:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
        else:
            in_weights, out_weights = None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))

    return dataset, in_splits, out_splits


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""

    def __init__(self, underlying_dataset, keys,args=None):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.transforms = {}
        self.data = args.dataset
        self.direct_return = isinstance(underlying_dataset, _SplitDataset)

    def __getitem__(self, key):
        if self.direct_return:
            return self.underlying_dataset[self.keys[key]]

        x, y = self.underlying_dataset[self.keys[key]]
        ret = {"y": y}

        for key, transform in self.transforms.items():
            ret[key] = transform(x)
        return ret

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset, n, seed=0,args=None):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert n <= len(dataset)
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1,args), _SplitDataset(dataset, keys_2,args)
