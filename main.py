import utils
from models import create_model
from dataset import create_dataset, dataset_convert_to_valid
import unlearn
from trainer import validate
from evaluation import get_membership_attack_prob, get_js_divergence, get_SVC_MIA
from attack import Attacker

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
    batch_size: int = 32
    seed: int = 42


def main(
    unlearn_name,
    dataset_name,
    task,
    model_name,
    input_size=32,
    batch_size=32,
    unlearn_labels=[0],
    num_classes=10,
    checkpoint="",
    retrain_checkpoint=None,
    seed=42,
):
    dataset = create_dataset(dataset_name, task, "./data", input_size)
    save_path = f"results/{unlearn_name}/{task}/{model_name}_{dataset_name}"
    os.makedirs(save_path, exist_ok=True)
    trainset, testset = dataset.get_datasets()
    transform_valid = testset.transform
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size, shuffle=True, num_workers=4)
    forget_trainset, retain_trainset, forget_validset, retain_validset = (
        dataset.full_class_split(unlearn_labels)
    )
    forget_trainloader = DataLoader(
        forget_trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    retain_trainloader = DataLoader(
        retain_trainset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    forget_valloader = DataLoader(
        forget_validset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    retain_valloader = DataLoader(
        retain_validset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    unlearn_dataloaders = OrderedDict(
        forget_train=forget_trainloader,
        retain_train=retain_trainloader,
        forget_valid=forget_valloader,
        retain_valid=retain_valloader,
        train=trainloader,
    )

    model = create_model(model_name, num_classes)
    model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.cuda()

    if retrain_checkpoint is not None and os.path.exists(retrain_checkpoint):
        retrain_model = create_model(model_name, num_classes)
        retrain_model.load_state_dict(
            torch.load(retrain_checkpoint)["state_dict"], strict=False
        )
        retrain_model.cuda()
    else:
        retrain_model = None

    loss_fn = nn.CrossEntropyLoss()
    args = Args(
        dataset=dataset_name,
        model=model_name,
        num_classes=num_classes,
        seed=seed,
        batch_size=batch_size,
    )

    start_time = time.time()
    unlearn_method = unlearn.create_unlearn_method(unlearn_name)(
        model, loss_fn, save_path, args
    )
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    unlearn_model = unlearn_method.get_unlearned_model()
    end_time = time.time()
    print("Num steps:", model.counter)

    time_elapsed = end_time - start_time
    torch.save(unlearn_model.state_dict(), os.path.join(save_path, "unlearn_model.pt"))

    evaluation_results = {"method": unlearn_name, "seed": seed, "steps": model.counter}
    for name, loader in unlearn_dataloaders.items():
        if loader is None:
            continue
        dataset_convert_to_valid(loader, transform_valid)
        eval_metrics = validate(loader, unlearn_model, loss_fn, name)
        for metr, v in eval_metrics.items():
            evaluation_results[f"{name}_{metr}"] = v

    attacker = Attacker(model, trainloader, forget_trainloader, testloader)
    evaluation_results["asr"] = attacker.attack()
    evaluation_results["iter"] = model.counter

    with open(os.path.join(save_path, f"results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == "__main__":
    for method in [
        "SFRon",
        "BadTeacher",
        "GradAscent",
        "RandomLabel",
        "SCRUB",
        "SalUn",
        "Finetune",
    ]:
        if method == "GradAscent":
            batch_size = 256
        elif method == "SalUn" or method == "SFRon":
            batch_size = 128
        else:
            batch_size = 32

        main(
            method,
            "CIFAR10",
            "FullClassUnlearn",
            "ResNet18",
            checkpoint="/home/owais/Machine_Unlearning/logs_1/CIFAR10/train/best_model.ckpt",
            # retrain_checkpoint="/home/owais/Machine_Unlearning/logs/CIFAR10/retrain_best/best_model.ckpt",
            unlearn_labels=[1],
            batch_size=batch_size,
        )
