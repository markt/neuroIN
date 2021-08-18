from torch.utils.data.dataset import random_split
from neuroIN.io.dataset import Dataset
from neuroIN.models.cnn2d import CNN2D

import os
import argparse
import numpy as np
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from functools import partial

from ray import tune
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB


def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            loss += criterion(outputs, target).item()

    return loss / len(data_loader), correct / total

def test_best_model(best_trial):
    test_loader = Dataset(best_trial.config["data_dir"]).test.get_dataloader(best_trial.config["batch_size"])

    best_model = best_trial.config["model"](n_classes=3, dropout_p=best_trial.config["dropout_p"],
        temp_kernel_length=best_trial.config["temp_kernel_length"],
        spac_kernel_length=best_trial.config["spac_kernel_length"],
        F1=best_trial.config["F1"],
        D=best_trial.config["D"],
        F2=best_trial.config["F2"],
        l2=best_trial.config["l2"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.value, "checkpoint")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_model.load_state_dict(model_state)

    mean_loss, mean_acc = test(best_model, test_loader)
    print(f"Test set has {mean_acc}% accuracy and mean loss of {mean_loss}")


def train_dataset(config, checkpoint_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config["model"](n_classes=3, dropout_p=config["dropout_p"],
        temp_kernel_length=config["temp_kernel_length"],
        spac_kernel_length=config["spac_kernel_length"],
        F1=config["F1"],
        D=config["D"],
        F2=config["F2"],
        l2=config["l2"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    if checkpoint_dir:
        config["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    dataset = Dataset(config["data_dir"])
    trainset = dataset.train

    test_abs = int(len(trainset) * .8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
    
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True)

    for epoch in range(36):
        train(model, optimizer, train_loader)
        mean_loss, acc = test(model, val_loader)

        tune.report(mean_accuracy=acc, mean_loss=mean_loss)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path)
    print("Finished Training")
    


def run_optim(config, max_concurrent=4, num_samples=40):
    if not isinstance(config, dict):
        config = torch.load(config)

    assert isinstance(config, dict), "'config' must be a dictionary"

    algo = TuneBOHB(max_concurrent=max_concurrent, metric="mean_loss", mode="min")
    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="mean_loss",
        mode="min",
        max_t=100)

    analysis = tune.run(train_dataset,
        num_samples=num_samples,
        config=config,
        scheduler=bohb,
        search_alg=algo)
    
    best_trial = analysis.get_best_trial('mean_loss', "min", "last")

    print(f"{best_trial.config}, {best_trial.last_result['mean_loss']}, {best_trial.last_result['mean_accuracy']}")
    test_best_model(best_trial)

def main():
    parser = argparse.ArgumentParser(description="Optimize a Dataset")
    parser.add_argument('data_dir', help="Directory of Dataset")
    args = parser.parse_args()
    print(args.data_dir)
    print(Dataset(args.data_dir))

    config = {
        "data_dir": Path(os.getcwd()) / args.data_dir,
        "checkpoint_dir": Path(os.getcwd()) / args.data_dir / "checkpoints",
        "model": CNN2D,
        "dropout_p": tune.uniform(0, 0.8),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.1, 0.9),
        "batch_size": tune.choice([8, 16]),
        "temp_kernel_length": tune.randint(64, 256),
        "spac_kernel_length": tune.randint(8, 24),
        "F1": tune.randint(4, 12),
        "D": tune.choice([1, 2, 4]),
        "F2": tune.randint(4, 12),
        "l2": tune.randint(1, 64)
    }
    run_optim(config)

if __name__ == "__main__":
    main()