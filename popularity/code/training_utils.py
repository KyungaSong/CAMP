import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def train(model, data_loader, optimizer, device, rank):
    model.train()
    total_loss = 0
    criteria = nn.MSELoss()

    if rank == 0:
        pbar = tqdm(data_loader, desc="Training")
    else:
        pbar = data_loader

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        pop_history_output, time_output, sideinfo_output, output = model(batch)

        target_index = batch['time'].long().unsqueeze(1) 
        pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
        # qlt_gt = batch['avg_rating'] / 5.0

        loss = criteria(output.squeeze(), pop_gt)
        # loss_q = criteria(sideinfo_output.squeeze(), qlt_gt)
        # loss = loss_p + loss_q

        loss.backward()
        optimizer.step()
        total_loss += loss.item() 

        torch.cuda.empty_cache()

    average_loss = total_loss / len(data_loader)
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, device, rank):
    model.eval()
    total_loss = 0
    criteria = nn.MSELoss()

    if rank == 0:
        pbar = tqdm(data_loader, desc="Evaluating")
    else:
        pbar = data_loader

    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pop_history_output, time_output, sideinfo_output, output = model(batch)
            
            target_index = batch['time'].long().unsqueeze(1) 
            pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
            # qlt_gt = batch['avg_rating'] / 5.0

            # loss_p = criteria(output.squeeze(), pop_gt)
            # loss_q = criteria(sideinfo_output.squeeze(), qlt_gt)
            # loss = loss_p + loss_q
            loss = criteria(output.squeeze(), pop_gt)

            total_loss += loss.item() 

            torch.cuda.empty_cache()

    average_loss = total_loss / len(data_loader)
    return average_loss

# 4) Testing
def test(model, data_loader, device, rank):
    model.eval()
    total_loss = 0
    total_rmse = 0
    criteria = nn.MSELoss()

    if rank == 0:
        pbar = tqdm(data_loader, desc="Testing")
    else:
        pbar = data_loader
    
    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            pop_history_output, time_output, sideinfo_output, output = model(batch)

            target_index = batch['time'].long().unsqueeze(1)
            pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
            # qlt_gt = batch['avg_rating'] / 5.0

            # loss_p = criteria(output.squeeze(), pop_gt)
            # loss_q = criteria(sideinfo_output.squeeze(), qlt_gt)
            # loss = loss_p + loss_q
            loss = criteria(output.squeeze(), pop_gt)

            total_loss += loss.item()

            torch.cuda.empty_cache()

            mse = loss.item()
            rmse = np.sqrt(mse)
            total_rmse += rmse

    average_loss = total_loss / len(data_loader)
    average_rmse = total_rmse / len(data_loader)
    return average_loss, average_rmse

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True