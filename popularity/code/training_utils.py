import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

import torch.cuda.amp as amp

def train(config, model, data_loader, optimizer, device):
    model.train()
    total_loss = 0
    criteria = nn.MSELoss()
    scaler = amp.GradScaler()  

    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        with amp.autocast():  
            pop_history_output, time_output, sideinfo_output, output = model(batch)

            target_index = batch['time'].long().unsqueeze(1)
            pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
            # print("pop_gt", pop_gt)
            avg_rating = batch['average_rating']            
            scaled_avg_rating = avg_rating * (pop_gt / 5.0)
            # print("scaled_avg_rating", scaled_avg_rating)

            loss_p = criteria(pop_history_output.squeeze(), pop_gt)
            loss_t = criteria(time_output.squeeze(), pop_gt)
            loss_s = criteria(sideinfo_output.squeeze(), scaled_avg_rating)
            loss_o = criteria(output.squeeze(), pop_gt)
            loss = config.wt_pop * loss_p + config.wt_time * loss_t + config.wt_side * loss_s + loss_o

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        torch.cuda.empty_cache()

    average_loss = total_loss / len(data_loader)
    
    return average_loss

def evaluate(config, model, data_loader, device):
    model.eval()
    total_loss = 0
    total_rmse = 0
    criteria = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            pop_history_output, time_output, sideinfo_output, output = model(batch)

            target_index = batch['time'].long().unsqueeze(1)
            pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
            avg_rating = batch['average_rating']
            scaled_avg_rating = avg_rating * (pop_gt / 5.0)

            loss_p = criteria(pop_history_output.squeeze(), pop_gt)
            loss_t = criteria(time_output.squeeze(), pop_gt)
            loss_s = criteria(sideinfo_output.squeeze(), scaled_avg_rating)
            loss_o = criteria(output.squeeze(), pop_gt)
            loss = config.wt_pop * loss_p + config.wt_time * loss_t + config.wt_side * loss_s + loss_o
            # loss = criteria(output.squeeze(), pop_gt)

            total_loss += loss.item()
            torch.cuda.empty_cache()

            mse = loss.item()
            rmse = np.sqrt(mse)
            total_rmse += rmse

    average_loss = total_loss / len(data_loader)
    average_rmse = total_rmse / len(data_loader)
    
    return average_loss, average_rmse

def test(config, model, test_loader, device):
    model.eval()
    test_loss = 0.0
    test_rmse = 0.0
    criteria = nn.MSELoss()

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}  

            pop_history_output, time_output, sideinfo_output, output = model(batch)

            target_index = batch['time'].long().unsqueeze(1)
            pop_gt = torch.gather(batch['pop_history'], 1, target_index).squeeze().float()
            
            avg_rating = batch['average_rating']
            scaled_avg_rating = avg_rating * (pop_gt / 5.0)
            
            loss_p = criteria(pop_history_output.squeeze(), pop_gt)
            loss_t = criteria(time_output.squeeze(), pop_gt)
            loss_s = criteria(sideinfo_output.squeeze(), scaled_avg_rating)
            loss_o = criteria(output.squeeze(), pop_gt)
            loss = config.wt_pop * loss_p + config.wt_time * loss_t + config.wt_side * loss_s + loss_o
            
            test_loss += loss.item()
            torch.cuda.empty_cache()

            rmse = torch.sqrt(loss).item()
            test_rmse += rmse

    average_loss = test_loss / len(test_loader)
    average_rmse = test_rmse / len(test_loader)
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
