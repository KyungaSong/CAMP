import random
import pickle
import logging
import os
import gc  
from datetime import datetime
import pandas as pd
import argparse
from tqdm import tqdm
import itertools

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist

from config import Config
from data_encoding import data_encoding
from preprocess import load_df, create_datasets
from Model import PopPredict
from training_utils import train, evaluate, test, EarlyStopping

########################################################### config
random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.7,
                    help="parameter for balance of pop_history and time")
parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=200,
                    help="training epochs")
parser.add_argument("--dataset", type=str, default='14_Sports',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", action="store_true",
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument("--embedding_dim", type=int, default=64,
                    help="embedding size for embedding vectors")
parser.add_argument("--wt_pop", type=float, default=0.1,
                    help="Weight parameter for balancing the loss contribution of pop_history")
parser.add_argument("--wt_time", type=float, default=1,
                    help="Weight parameter for balancing the loss contribution of release_time")
parser.add_argument("--wt_side", type=float, default=1,
                    help="Weight parameter for balancing the loss contribution of side_information")

parser.add_argument('--cuda_device', type=str, help='CUDA device to use')

args = parser.parse_args()
config = Config(args=args)

def setup_logging(dataset_name):
    log_dir = os.path.abspath('./log')
    dataset_log_dir = os.path.join(log_dir, dataset_name)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    current_date = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(dataset_log_dir, f'{current_date}_log.txt')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d')

def generate_outputs(model, data_loader, device):
    model.eval()
    all_outputs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating Outputs"):
            batch = {k: v.to(device) for k, v in batch.items()}
            weighted_pop_history_output, weighted_time_output, weighted_sideinfo_output, output = model(batch)
            for i in range(len(batch['item'])):
                all_outputs.append({
                    'weighted_pop_history_output': weighted_pop_history_output[i].cpu().numpy(),
                    'weighted_time_output': weighted_time_output[i].cpu().numpy(),
                    'weighted_sideinfo_output': weighted_sideinfo_output[i].cpu().numpy()
                })

    return all_outputs

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    setup_logging(config.dataset)

    if not os.path.exists(config.review_file_path):
        data_encoding(config)
    else:
        print(f"{config.review_file_path} already exists.")

    best_loss = float('inf')
    best_model_params = {}

    train_df, valid_df, test_df, combined_df, num_items, num_cats, num_stores, max_time = load_df(config)
    gc.collect()
    
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
    del train_df, valid_df, test_df
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr_values = [0.001, 0.005, 0.01]  # Learning rates to try
    batch_size_values = [32, 64]  # Batch sizes to try
    embedding_dim_values = [64, 128]  # Embedding dimensions to try
    # lr_values = [0.01]  # Learning rates to try
    # batch_size_values = [32]  # Batch sizes to try
    # embedding_dim_values = [768]  # Embedding dimensions to try

    # # Toys_and_Games
    # lr: 0.001, batch_size: 16, embedding_dim: 64
    # Sports_and_Outdoors
    # Best Model Parameters: {'lr': 0.0001, 'batch_size': 16, 'embedding_dim': 128, 'epoch': 0}

    best_loss = float('inf')
    best_model_params = {}

    for lr, batch_size, embedding_dim in itertools.product(lr_values, batch_size_values, embedding_dim_values): 
        print(f"Training with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")
        
        config.lr = lr
        config.batch_size = batch_size
        config.embedding_dim = embedding_dim
        
        model = PopPredict(config, num_items, num_cats, num_stores, max_time).to(device)
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        for epoch in range(config.num_epochs):
            train_loss = train(config, model, train_loader, optimizer, device)
            valid_loss, valid_rmse = evaluate(config, model, valid_loader, device)
            scheduler.step() 

            torch.cuda.empty_cache()
            
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid RMSE: {valid_rmse:.4f}')

            if valid_loss < best_loss:
                logging.info(f'Best model is changed in epoch {epoch+1}')
                best_loss = valid_loss
                best_model_params = {'lr': config.lr, 'batch_size': config.batch_size, 'embedding_dim': config.embedding_dim, 'epoch': epoch + 1}
                best_model = model.state_dict()
            
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        test_loss, test_rmse = test(config, model, test_loader, device)
        logging.info(f'Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}_with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}')
        print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}_with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")

    if best_model is not None:
        print(f"Best Model Parameters: {best_model_params}")
        logging.info(f"Best Model Parameters: {best_model_params}")
        date_str = datetime.now().strftime('%y%m%d')
        config.embedding_dim = best_model_params['embedding_dim']
        model = PopPredict(config, num_items, num_cats, num_stores, max_time).to(device)
        model.load_state_dict(best_model)
        final_save_path = f'{config.model_save_path}{date_str}_best_model.pt'
        if not os.path.exists(os.path.dirname(config.model_save_path)):
            os.makedirs(os.path.dirname(config.model_save_path))
        torch.save({
            'model_state_dict': model.state_dict(),
            'embedding_dim': best_model_params['embedding_dim'],  # Save the best embedding dimension
            'lr': best_model_params['lr'],
            'batch_size': best_model_params['batch_size']
        }, final_save_path)
        
        test_loss, test_rmse = test(config, model, test_loader, device)
        logging.info(f'Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}')
        print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}")

    pred_dataset = create_datasets(combined_df, combined_df, combined_df)[2]
    pred_loader = DataLoader(pred_dataset, batch_size=config.batch_size, shuffle=False)

    outputs = generate_outputs(model, pred_loader, device)
    outputs_df = pd.DataFrame(outputs)

    results_df = pd.concat([combined_df.reset_index(drop=True), outputs_df], axis=1)

    results_df['time_output'] = results_df['weighted_time_output'].apply(lambda x: x[0])
    results_df['conformity'] = results_df['weighted_pop_history_output'].apply(lambda x: x[0]) + results_df['weighted_time_output'].apply(lambda x: x[0])
    results_df['quality'] = results_df['weighted_sideinfo_output'].apply(lambda x: x[0])
    results_df = results_df[['item_encoded', 'unit_time', 'time_output', 'conformity', 'quality']]

    result_path = f'{config.dataset_path}pop_{config.dataset}.pkl'
    results_df.to_pickle(result_path)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()
