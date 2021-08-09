from loaddata import load_train_test
from models import CNN2D

from datetime import datetime
from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_loop(dataloader, model, loss_fn, optimizer, device="cpu", verbose=False):
    size = len(dataloader.dataset)
    
    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            loss, samples = loss.item(), i * len(X)
            if verbose: print(f"[{samples:>3d}/{size:>3d}] loss: {loss:>7f}")

def test_loop(dataloader, model, loss_fun, device="cpu", name='Test', verbose=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            test_loss += loss_fun(y_pred, y).item()
            correct += (y_pred.argmax(1) == y).float().sum().item()
    
    test_loss = test_loss / num_batches
    correct = correct / size
    if verbose: print(f"{name} Error: Accuracy {(100*correct):>0.1f}, Avg loss: {test_loss:>.7f}")
    
    return correct, test_loss

def train(data_dir, epochs=10, batch_size=24, lr=1e-2, momentum=0.9, dropout_p=0.5, use_GPU=True, shuffle=True, num_channels=64, net=None, verbose=False):
    init_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")
    data_path = Path(data_dir)

    train_loader, val_loader, test_loader = [DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in load_train_test(data_dir)]
    train_df = train_loader.dataset.df

    n_classes = train_df['label'].nunique()
    # net = CNN2D(n_classes=n_classes, dropout_p=dropout_p, num_channels=num_channels)
    if not net: net = CNN2D(n_classes=n_classes, dropout_p=dropout_p, num_channels=num_channels)
    net = net.to(device)

    if 'weights' in train_df:
        weights = train_df.weights
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    hist_d = {'Train_acc': [], 'Train_loss': [], 'Val_acc': [], 'Val_loss': []}
    best_val_acc = 0
    epochs_since_improv = 0
    improv_thresh = 20
    for i in range(epochs):
        if verbose: print(f"\nEpoch {i+1}\n---------------------")
        train_loop(train_loader, net, criterion, optimizer, device=device, verbose=verbose)
        train_acc, train_loss = test_loop(train_loader, net, criterion, device=device, name='Train', verbose=verbose)
        val_acc, val_loss = test_loop(val_loader, net, criterion, device=device, name='Val', verbose=verbose)

        hist_d['Train_acc'].append(train_acc)
        hist_d['Train_loss'].append(train_loss)
        hist_d['Val_acc'].append(val_acc)
        hist_d['Val_loss'].append(val_loss)

        if val_acc > best_val_acc:
            torch.save(net.state_dict(), data_path / f'checkpoint_{init_timestamp}.pt')
            best_val_acc = val_acc
            print(f'Checkpoint saved at epoch {i}')
            epochs_since_improv = 0

        epochs_since_improv += 1
        if epochs_since_improv > improv_thresh:
            print(f'val accuracy has not improved in {improv_thresh} epochs, stopping early')
            break
    history = pd.DataFrame(data=hist_d)

    print(f'Model params: epochs={epochs}, batch_size={batch_size}, lr={lr}, momentum={momentum}, dropout_p={dropout_p}')
    print(f'Best validation accuracy was {(100*best_val_acc):>0.1f}%')

    test_loop(test_loader, net, criterion, device=device, verbose=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    data_path = Path(data_dir)
    torch.save(net.state_dict(), data_path / f'model_{timestamp}.pt')
    history.to_csv(data_path / f'history_{timestamp}.csv')