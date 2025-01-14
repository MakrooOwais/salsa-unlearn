import utils
from models import create_model
from dataset import create_dataset, dataset_convert_to_valid
import unlearn
from trainer import validate
from evaluation import get_membership_attack_prob, get_js_divergence, get_SVC_MIA


import os
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch
from torch import nn
import time
from dataclasses import dataclass
import json


@dataclass
class Args:
    dataset: str = ""
    model: str = ""
    num_classes: int = 10
    seed: int = 42


def main(
    unlearn_name,
    dataset_name,
    task,
    model_name,
    input_size=32,
    batch_size=32,
    forget_perc=0.1,
    num_classes=10,
    checkpoint="",
    retrain_checkpoint=None,
    seed=42,
):
    dataset = create_dataset(dataset_name, task, "./data", input_size)
    save_path = f"results/{task}/{model_name}_{dataset_name}"
    os.makedirs(save_path, exist_ok=True)
    trainset, testset = dataset.get_dataset()
    transform_valid = testset.transform
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=4)
    forget_trainset, retain_trainset = dataset.random_split(forget_perc, save_path)
    print(len(forget_trainset), len(retain_trainset))
    forget_trainloader = DataLoader(
        forget_trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    retain_trainloader = DataLoader(
        retain_trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    unlearn_dataloaders = OrderedDict(
        forget_train=forget_trainloader,
        retain_train=retain_trainloader,
        forget_valid=None,
        retain_valid=testloader,
        train=trainloader,
    )

    model = create_model(model_name, num_classes)
    model.load_state_dict(torch.load(checkpoint))
    model.cuda()

    if retrain_checkpoint is not None and os.path.exists(retrain_checkpoint):
        retrain_model = create_model(model_name, num_classes)
        retrain_model.load_state_dict(torch.load(retrain_checkpoint))
        retrain_model.cuda()
    else:
        retrain_model = None

    loss_fn = nn.CrossEntropyLoss()
    args = Args(
        dataset=dataset_name, model=model_name, num_classes=num_classes, seed=seed
    )

    start_time = time.time()
    unlearn_method = unlearn.create_unlearn_method(unlearn_name)(
        model, loss_fn, save_path, args
    )
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    unlearn_model = unlearn_method.get_unlearn_model()
    end_time = time.time()
    time_elapsed = end_time - start_time

    evaluation_results = {"method": unlearn_method, "seed": seed}
    for name, loader in unlearn_dataloaders.items():
        if loader is None:
            continue
        dataset_convert_to_valid(loader, transform_valid)
        eval_metrics = validate(loader, unlearn_model, loss_fn, name)
        for metr, v in eval_metrics.items():
            evaluation_results[f"{name}_{metr}"] = v

    for mia_metric in ["entropy"]:
        evaluation_results[f"{mia_metric}_mia"] = get_membership_attack_prob(
            retain_trainloader,
            forget_trainloader,
            testloader,
            unlearn_model,
            mia_metric,
        )

    dataset_convert_to_valid(retain_trainset, transform_valid)
    if retrain_model:
        evaluation_results["js_div"], evaluation_results["kl_div"] = get_js_divergence(
            forget_trainloader, unlearn_model, retrain_model
        )
    else:
        evaluation_results["js_div"], evaluation_results["kl_div"] = None, None

    evaluation_results["time"] = time_elapsed
    evaluation_results["params"] = unlearn_method.get_params()

    with open(os.path.join(save_path, f"results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == "__main__":
    main()