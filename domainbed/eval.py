import collections
import copy
import json
import time
from pathlib import Path

import loralib as lora
import numpy as np
import torch
import torch.utils.data

from domainbed import algorithms
from domainbed import swad as swad_module
from domainbed.datasets import get_dataset, split_dataset
from domainbed.evaluator import Evaluator
from domainbed.lib import misc, swa_utils
from domainbed.lib.fast_data_loader import FastDataLoader, InfiniteDataLoader
from domainbed.lib.query import Q


def json_handler(v):
    if isinstance(v, (Path, range)):
        return str(v)
    raise TypeError(f"`{type(v)}` is not JSON Serializable")

def mark_only_attn_as_trainable(model) -> None:
    for n, p in model.named_parameters():
        if 'attn' not in n:
            p.requires_grad = False

def eval_en(test_envs, args, hparams, n_steps, checkpoint_freq, logger, writer, target_env=None,sweep=False):
    logger.info("")

    #######################################################
    # setup dataset & loader
    #######################################################
    args.real_test_envs = test_envs  # for log
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    dataset, in_splits, out_splits = get_dataset(test_envs, args, hparams, algorithm_class)
    test_splits = []

    if target_env is not None:
        testenv_name = f"te_{dataset.environments[target_env]}"
        logger.info(f"Target env = {target_env}")
    else:
        testenv_properties = [str(dataset.environments[i]) for i in test_envs]
        testenv_name = "te_" + "_".join(testenv_properties)

    logger.info(
        "Testenv name escaping {} -> {}".format(testenv_name, testenv_name.replace(".", ""))
    )
    testenv_name = testenv_name.replace(".", "")
    logger.info(f"Test envs = {test_envs}, name = {testenv_name}")

    n_envs = len(dataset)
    train_envs = sorted(set(range(n_envs)) - set(test_envs))
    iterator = misc.SplitIterator(test_envs)
    batch_sizes = np.full([n_envs], hparams["batch_size"], dtype=int)

    batch_sizes[test_envs] = 0
    batch_sizes = batch_sizes.tolist()

    logger.info(f"Batch sizes for each domain: {batch_sizes} (total={sum(batch_sizes)})")

    # calculate steps per epoch
    steps_per_epochs = [
        len(env) / batch_size
        for (env, _), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]
    steps_per_epoch = min(steps_per_epochs)
    # epoch is computed by steps_per_epoch
    prt_steps = ", ".join([f"{step:.2f}" for step in steps_per_epochs])
    logger.info(f"steps-per-epoch for each domain: {prt_steps} -> min = {steps_per_epoch:.2f}")
    best_acc=0.
    # setup loaders
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=batch_size,
            num_workers=dataset.N_WORKERS,
        )
        for (env, env_weights), batch_size in iterator.train(zip(in_splits, batch_sizes))
    ]

    # setup eval loaders
    eval_loaders_kwargs = []
    for i, (env, _) in enumerate(in_splits + out_splits + test_splits):
        batchsize = hparams["test_batchsize"]
        loader_kwargs = {"dataset": env, "batch_size": batchsize, "num_workers": dataset.N_WORKERS}
        if args.prebuild_loader:
            loader_kwargs = FastDataLoader(**loader_kwargs)
        eval_loaders_kwargs.append(loader_kwargs)

    eval_weights = [None for _, weights in (in_splits + out_splits + test_splits)]
    eval_loader_names = ["env{}_in".format(i) for i in range(len(in_splits))]
    eval_loader_names += ["env{}_out".format(i) for i in range(len(out_splits))]
    eval_loader_names += ["env{}_inTE".format(i) for i in range(len(test_splits))]
    eval_meta = list(zip(eval_loader_names, eval_loaders_kwargs, eval_weights))
    # breakpoint()
    #######################################################
    # setup algorithm (model)
    #######################################################
    algorithms_= []
    import os
    sd_list = os.listdir(f'./ensemble/{args.dataset}')
    for i in range(len(sd_list)):
        sd_list[i]=f'./ensemble/{args.dataset}/'+sd_list[i]+f'/checkpoints/TE{test_envs[0]}.pth'
    print(sd_list)
    for i in range(3):
        algorithms_.append(algorithm_class(
            dataset.input_shape,
            dataset.num_classes,
            len(dataset) - len(test_envs),
            hparams,
            args
        ))
        t=torch.load(sd_list[i])
        # breakpoint()
        algorithms_[-1].load_state_dict(t['model_dict'])
        algorithms_[-1].cuda().eval()
        print(f"loaded {i}th model")
    
    
    # n_params = sum([p.numel() for p in algorithm.parameters()])
    # logger.info("# of params = %d" % n_params)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    #######################################################
    # start training loop
    #######################################################
    evaluator = Evaluator(
        test_envs,
        eval_meta,
        n_envs,
        logger,
        evalmode=args.evalmode,
        debug=args.debug,
        target_env=target_env,
    )


    last_results_keys = None
    records = []
    epochs_path = args.out_dir / "results.jsonl"

    step=0
    results = {
        "step": step,
        "epoch": step / steps_per_epoch,
    }

    for key, val in checkpoint_vals.items():
        results[key] = np.mean(val)

    eval_start_time = time.time()
    accuracies, summaries = evaluator.evaluate_en(algorithms_)
    results["eval_time"] = time.time() - eval_start_time

    results_keys = list(summaries.keys()) + sorted(accuracies.keys()) + list(results.keys())
    # merge results
    results.update(summaries)
    results.update(accuracies)
    
    # print
    if results_keys != last_results_keys:
        logger.info(misc.to_row(results_keys))
        last_results_keys = results_keys
    logger.info(misc.to_row([results[key] for key in results_keys]))
    records.append(copy.deepcopy(results))

    # update results to record
    results.update({"hparams": dict(hparams), "args": vars(args)})

    with open(epochs_path, "a") as f:
        f.write(json.dumps(results, sort_keys=True, default=json_handler) + "\n")

    checkpoint_vals = collections.defaultdict(lambda: [])

    writer.add_scalars_with_prefix(summaries, step, f"{testenv_name}/summary/")
    writer.add_scalars_with_prefix(accuracies, step, f"{testenv_name}/all/")
    


    # find best
    logger.info("---")
    records = Q(records)
    te_val_best = records.argmax("test_out")["test_in"]
    tr_val_best = records.argmax("train_out")["test_in"]
    last = records[-1]["test_in"]

    in_key = "train_out"
    tr_val_best_indomain = records.argmax("train_out")[in_key]
    last_indomain = records[-1][in_key]

    # NOTE for clearity, report only training-domain validation results.
    ret = {
        #  "test-domain validation": te_val_best,
        "training-domain validation": tr_val_best,
        #  "last": last,
        #  "last (inD)": last_indomain,
        #  "training-domain validation (inD)": tr_val_best_indomain,
    }

            
    for k, acc in ret.items():
        logger.info(f"{k} = {acc:.3%}")
    if sweep:
        return ret, records, best_acc
    return ret, records
