import random
import pickle
import time
import logging
import os
import time
from datetime import datetime
import pandas as pd
import argparse

from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets
from Model import ModulePopHistory, PopPredict
from training_utils import train, evaluate, test


########################################################### config
random.seed(2024) 
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.7,
                    help="parameter for balance of pop_history and time")
parser.add_argument("--batch_size", type=int, default=128,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="training epochs")
parser.add_argument("--time_unit", type=int, default=1000 * 60 * 60 * 24,
                    help="smallest time unit for model training")
parser.add_argument("--pop_time_unit", type=int, default=30,
                    help="smallest time unit for item popularity statistic")
parser.add_argument("--data_preprocessed", type=bool, default=True,
                    help="whether the input data has already been preprocessed")
parser.add_argument("--embedding_dim", type=int, default=512,
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

    if os.path.exists(f'{processed_path}/train_df.pkl') and os.path.exists(f'{processed_path}/valid_df.pkl') and os.path.exists(f'{processed_path}/test_df.pkl') and config.data_preprocessed:
        with open(f'{processed_path}/train_df.pkl', 'rb') as file:
            train_df = pickle.load(file)
        with open(f'{processed_path}/valid_df.pkl', 'rb') as file:
            valid_df = pickle.load(file)
        with open(f'{processed_path}/test_df.pkl', 'rb') as file:
            test_df = pickle.load(file)
        
        combined_df = pd.concat([train_df, valid_df, test_df])
        num_users = combined_df['user_encoded'].nunique()
        num_items = combined_df['item_encoded'].nunique()
        num_cats = combined_df['cat_encoded'].nunique()
        num_stores = combined_df['store_encoded'].nunique()
        print("Processed files already exist. Skipping dataset preparation.")
    else:
        try:
            start_time = time.time()
            df = load_dataset(review_file_path)
            df_meta = load_dataset(meta_file_path, meta=True)
            
            num_users = df['user_id'].nunique()
            num_items = df['item_id'].nunique()
            num_cats = df_meta['category'].nunique()
            num_stores = df_meta['store'].nunique()
            print(f'num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}, num_stores: {num_stores}')

            df = pd.merge(df, df_meta, on='item_id', how='left')
            print("df:\n", df)
            train_df, valid_df, test_df, max_time = preprocess_df(df, config)

            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%Y%m%d')
            train_df.to_pickle(f'{processed_path}/train_df_pop_{date_str}_{num_users}_{num_items}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_pop_{date_str}_{num_users}_{num_items}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_pop_{date_str}_{num_users}_{num_items}.pkl')

            end_time = time.time()
            logging.info(f"Dataset prepared in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during data preparation: {str(e)}")
            raise

    return train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time

def main():
    print("Data preprocessing......")
    setup_logging()
    dataset_name = 'sampled_Home_and_Kitchen'
    train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time = load_data(dataset_name = dataset_name)

    print("Create datasets......")
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)

    print("Making Data loader......")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    print("Training......")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PopPredict(True, config, num_items, max_time).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Training......")
    # Train and evaluate
    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        # valid_loss = evaluate(model, valid_loader, device)
        # logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

    # save_path = "../../model/"
    # os.makedirs(save_path, exist_ok=True)
    # model_filename = f"trained_pop_model_{datetime.now().strftime('%Y-%m-%d')}.pt"
    # full_path = os.path.join(save_path, model_filename)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'train_loss': train_loss,
    #     'valid_loss': valid_loss,
    #     'epoch': epoch,
    # }, full_path)
    # logging.info(f"Model and training states have been saved to {full_path}")

if __name__ == "__main__":
    main()