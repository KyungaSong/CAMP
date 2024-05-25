import random
import pickle
import logging
import os
import gc  
from datetime import datetime
import pandas as pd
import argparse
import itertools

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.1,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=50,
                    help="training epochs")
parser.add_argument("--time_unit", type=int, default=1000 * 60 * 60 * 24,
                    help="smallest time unit for model training")
parser.add_argument("--pop_time_unit", type=int, default=30,
                    help="smallest time unit for item popularity statistic")
parser.add_argument("--dataset", type=str, default='sampled_Home_and_Kitchen',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", action="store_true",
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument("--embedding_dim", type=int, default=128,
                    help="embedding size for embedding vectors")

args = parser.parse_args()
config = Config(args=args)

def setup_logging():
    logging.basicConfig(filename='../../pop_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(dataset_name):
    dataset_path = f'../../dataset/{dataset_name}/'
    review_file_path = f'{dataset_path}{dataset_name}.pkl'
    meta_file_path = f'{dataset_path}meta_{dataset_name}.pkl'
    processed_path = f'../../dataset/preprocessed/pop/{dataset_name}/'

    combined_df = None  

    if os.path.exists(f'{processed_path}/train_df_pop.pkl') and os.path.exists(f'{processed_path}/valid_df_pop.pkl') and os.path.exists(f'{processed_path}/test_df_pop.pkl') and config.data_preprocessed:
        with open(f'{processed_path}/train_df_pop.pkl', 'rb') as file:
            train_df = pickle.load(file)
        with open(f'{processed_path}/valid_df_pop.pkl', 'rb') as file:
            valid_df = pickle.load(file)
        with open(f'{processed_path}/test_df_pop.pkl', 'rb') as file:
            test_df = pickle.load(file)

        combined_df = pd.concat([train_df, valid_df, test_df])
        num_users = combined_df['user_id'].nunique()
        num_items = combined_df['item_encoded'].nunique()
        num_cats = combined_df['cat_encoded'].nunique()
        num_stores = combined_df['store_encoded'].nunique()
        max_time = combined_df["unit_time"].max()
        print("Processed files already exist. Skipping dataset preparation.")
    else:
        try:
            df = load_dataset(review_file_path)
            df_meta = load_dataset(meta_file_path, meta=True)

            num_users = df['user_id'].nunique()
            num_items = df['item_id'].nunique()
            num_cats = df_meta['category'].nunique()
            num_stores = df_meta['store'].nunique()

            df = pd.merge(df, df_meta, on='item_id', how='left')
            train_df, valid_df, test_df, max_time = preprocess_df(df, config)

            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%Y%m%d')
            train_df.to_pickle(f'{processed_path}/train_df_pop_{date_str}_{num_users}_{num_items}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_pop_{date_str}_{num_users}_{num_items}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_pop_{date_str}_{num_users}_{num_items}.pkl')

            combined_df = pd.concat([train_df, valid_df, test_df])
        except Exception as e:
            raise

    if combined_df is not None:
        del combined_df
    if 'df' in locals():
        del df
    if 'df_meta' in locals():
        del df_meta
    gc.collect()

    return train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time

def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    setup_logging()

    learning_rates = [0.0001, 0.01, 0.1]
    batch_sizes = [32, 64]  # Reduced maximum batch size
    embedding_dims = [64, 128]

    best_rmse = float('inf')
    best_model_params = {}

    for lr, batch_size, embedding_dim in itertools.product(learning_rates, batch_sizes, embedding_dims):
        config.lr = lr
        config.batch_size = batch_size
        config.embedding_dim = embedding_dim

        train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time = load_data(config.dataset)
        gc.collect()
        
        train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
        del train_df, valid_df, test_df
        gc.collect()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = PopPredict(True, config, num_items, num_cats, num_stores, max_time).to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
        early_stopping = EarlyStopping(patience=10, verbose=True)

        if not config.test_only:
            for epoch in range(config.num_epochs):
                train_loss = train(model, train_loader, optimizer, device)
                valid_loss, valid_rmse = evaluate(model, valid_loader, device)
                scheduler.step(valid_loss)

                torch.cuda.empty_cache()  # Clear cache after each epoch

                logging.info(f'Epoch {epoch+1}, LR: {lr}, Batch Size: {batch_size}, Embed Dim: {embedding_dim}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid RMSE: {valid_rmse:.4f}')

                if valid_rmse < best_rmse:
                    best_rmse = valid_rmse
                    best_model_params = {'lr': lr, 'batch_size': batch_size, 'embedding_dim': embedding_dim, 'epoch': epoch}
                    torch.save(model.state_dict(), '../../model/pop/best_model.pt')

                if early_stopping(valid_loss):
                    print("Early stopping triggered.")
                    break

    logging.info(f'Best Model Parameters: {best_model_params} with RMSE: {best_rmse}')
    print(f"Best Model Parameters: {best_model_params} with RMSE: {best_rmse}")

if __name__ == "__main__":
    main()
