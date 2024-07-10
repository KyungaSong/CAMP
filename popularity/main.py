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
from preprocess import load_dataset, preprocess_df, create_datasets
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
parser.add_argument("--num_epochs", type=int, default=30,
                    help="training epochs")
parser.add_argument("--time_unit", type=int, default=1000*60*60*24,
                    help="smallest time unit for model training(default: day)")
parser.add_argument("--pop_time_unit", type=int, default=30*3,
                    help="smallest time unit for item popularity statistic")
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


args = parser.parse_args()
config = Config(args=args)

def setup_logging(dataset_name):
    log_dir = os.path.abspath('../../pop_log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    dataset_log_dir = os.path.join(log_dir, dataset_name)
    if not os.path.exists(dataset_log_dir):
        os.makedirs(dataset_log_dir)

    current_date = datetime.now().strftime('%Y-%m-%d')
    log_file = os.path.join(dataset_log_dir, f'{current_date}_log.txt')
    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        datefmt='%Y-%m-%d')

def load_data(dataset_name):
    dataset_path = f'../../dataset/{dataset_name}/'    
    sampled_file_path = f'{dataset_path}{dataset_name}.pkl'      
    processed_path = f'{dataset_path}preprocessed/'

    combined_df = None  

    if os.path.exists(f'{processed_path}/train_df_pop.pkl') and os.path.exists(f'{processed_path}/valid_df_pop.pkl') and os.path.exists(f'{processed_path}/test_df_pop.pkl') and config.data_preprocessed:
        with open(f'{processed_path}/train_df_pop.pkl', 'rb') as file:
            train_df = pickle.load(file)
        with open(f'{processed_path}/valid_df_pop.pkl', 'rb') as file:
            valid_df = pickle.load(file)
        with open(f'{processed_path}/test_df_pop.pkl', 'rb') as file:
            test_df = pickle.load(file)

        combined_df = pd.concat([train_df, valid_df, test_df])
        num_items = combined_df['item_encoded'].max() + 1
        num_cats = combined_df['cat_encoded'].max() + 1
        num_stores = combined_df['store_encoded'].max() + 1
        max_time = combined_df["unit_time"].max()
        print("Processed files already exist. Skipping dataset preparation.")
    else:
        try:
            if config.dataset[:8] == 'sampled_':
                review_file_path = f'../../dataset/{dataset_name[8:]}/{dataset_name[8:]}.pkl'
                df = load_dataset(review_file_path)
                sampled_df = load_dataset(sampled_file_path)
                sampled_items = sampled_df['item_encoded'].unique()
                filtered_df = df[df['item_encoded'].isin(sampled_items)]
            else:
                filtered_df = load_dataset(sampled_file_path)

            num_items = filtered_df['item_encoded'].max() + 1
            num_cats = filtered_df['cat_encoded'].max() + 1
            num_stores = filtered_df['store_encoded'].max() + 1

            train_df, valid_df, test_df, max_time = preprocess_df(filtered_df, config)
            combined_df = pd.concat([train_df, valid_df, test_df])
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%y%m%d%H%M')
            train_df.to_pickle(f'{processed_path}/train_df_pop_{date_str}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_pop_{date_str}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_pop_{date_str}.pkl')

        except Exception as e:
            raise

    if 'df' in locals():
        del df
    gc.collect()

    return train_df, valid_df, test_df, combined_df, num_items, num_cats, num_stores, max_time

def load_model_state(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    embedding_dim = checkpoint['embedding_dim']
    
    model.item_embedding = nn.Embedding(model.item_embedding.num_embeddings, embedding_dim).to(device)
    model.cat_embedding = nn.Embedding(model.cat_embedding.num_embeddings, embedding_dim).to(device)
    model.store_embedding = nn.Embedding(model.store_embedding.num_embeddings, embedding_dim).to(device)
    model.time_embedding = nn.Embedding(model.time_embedding.num_embeddings, embedding_dim).to(device)
        
    model.module_time.fc_time_value = nn.Linear(4 * embedding_dim, 1).to(device)
    model.module_sideinfo.fc_output = nn.Linear(2 * embedding_dim, 1).to(device)

    new_state_dict = {}
    for k, v in checkpoint['model_state_dict'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)

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

    best_loss = float('inf')
    best_model_params = {}
    model_save_path = None

    train_df, valid_df, test_df, combined_df, num_items, num_cats, num_stores, max_time = load_data(config.dataset)
    gc.collect()
    
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
    del train_df, valid_df, test_df
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lr_values = [0.001, 0.01]  # Learning rates to try
    batch_size_values = [32, 64]  # Batch sizes to try
    embedding_dim_values = [64, 128]  # Embedding dimensions to try

    # # Toys_and_Games
    # lr: 0.001, batch_size: 16, embedding_dim: 64
    # Sports_and_Outdoors
    # Best Model Parameters: {'lr': 0.0001, 'batch_size': 16, 'embedding_dim': 128, 'epoch': 0}

    best_loss = float('inf')
    best_model_params = {}
    best_model = None

    for lr, batch_size, embedding_dim in itertools.product(lr_values, batch_size_values, embedding_dim_values): 
        print(f"Training with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")
        
        config.lr = lr
        config.batch_size = batch_size
        config.embedding_dim = embedding_dim
        
        model = PopPredict(config, num_items, num_cats, num_stores, max_time).to(device)
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)
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
                best_model_params = {'lr': config.lr, 'batch_size': config.batch_size, 'embedding_dim': config.embedding_dim, 'epoch': epoch}
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
        config.embedding_dim = best_model_params['embedding_dim']
        model = PopPredict(config, num_items, num_cats, num_stores, max_time).to(device)
        model.load_state_dict(best_model)
        model_save_path = f'../../model/pop/{config.dataset}/best_model.pt'
        if not os.path.exists(os.path.dirname(model_save_path)):
            os.makedirs(os.path.dirname(model_save_path))
        torch.save({
            'model_state_dict': model.state_dict(),
            'embedding_dim': best_model_params['embedding_dim'],  # Save the best embedding dimension
            'lr': best_model_params['lr'],
            'batch_size': best_model_params['batch_size']
        }, model_save_path)
        
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

    result_path = f'../../dataset/{config.dataset}/pop_{config.dataset}.pkl'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results_df.to_pickle(result_path)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()
