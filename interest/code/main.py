import random
import pickle
import time
import logging
import os
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets
from Model import CAMP
from training_utils import train, evaluate, test

random.seed(2024) 
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

def setup_logging():
    logging.basicConfig(filename='../../log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(dataset_name):    
    dataset_path = f'../../dataset/{dataset_name}/'
    review_file_path = f'{dataset_path}{dataset_name}.pkl'
    meta_file_path = f'{dataset_path}meta_{dataset_name}.pkl'
    processed_path = f'../../dataset/preprocessed/{dataset_name}/'

    if os.path.exists(f'{processed_path}/train_df.pkl') and os.path.exists(f'{processed_path}/valid_df.pkl') and os.path.exists(f'{processed_path}/test_df.pkl') and Config.data_preprocessed:
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
        item_to_cat_dict = dict(zip(combined_df['item_encoded'], combined_df['cat_encoded']))
        print("Processed files already exist. Skipping dataset preparation.")
    else:
        try:
            start_time = time.time()
            df = load_dataset(review_file_path)
            df_meta = load_dataset(meta_file_path, meta=True)

            num_users = df['user_id'].nunique()
            num_items = df['item_id'].nunique()
            num_cats = df_meta['category'].nunique()
            print(f'num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}')
            df = pd.merge(df, df_meta, on='item_id', how='left')
            train_df, valid_df, test_df, item_to_cat_dict = preprocess_df(df, Config)

            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%Y%m%d')
            train_df.to_pickle(f'{processed_path}/train_df_{date_str}_{num_users}_{num_items}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_{date_str}_{num_users}_{num_items}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_{date_str}_{num_users}_{num_items}.pkl')

            end_time = time.time()
            logging.info(f"Dataset prepared in {end_time - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during data preparation: {str(e)}")
            raise

    return train_df, valid_df, test_df, num_users, num_items, num_cats, item_to_cat_dict

def main():
    print("Data preprocessing......")
    setup_logging()
    dataset_name = 'sampled_Home_and_Kitchen'
    train_df, valid_df, test_df, num_users, num_items, num_cats, item_to_cat_dict = load_data(dataset_name = dataset_name)

    print("Create datasets......")
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
    print("Making Data loader......")
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CAMP(num_users, num_items, num_cats, Config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        # print(f"Let's use 2 of the available GPUs!")
        # model = nn.DataParallel(model, device_ids=[0, 1])
        
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    print("Training......")
    # Train and evaluate
    for epoch in range(Config.num_epochs):
        train_loss = train(model, train_loader, optimizer, item_to_cat_dict, device)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, item_to_cat_dict, device)
        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Accuracy: {valid_accuracy}')

    save_path = "../../model/"
    os.makedirs(save_path, exist_ok=True)
    model_filename = f"trained_model_{datetime.now().strftime('%Y-%m-%d')}.pt"
    full_path = os.path.join(save_path, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'epoch': epoch,
    }, full_path)
    logging.info(f"Model and training states have been saved to {full_path}")

    # Evaluate on test set
    average_loss, all_top_k_items, avg_precision, avg_recall, avg_ndcg, avg_ndcg_2, avg_hit_rate, avg_auc, avg_mrr = test(model, test_loader, item_to_cat_dict, device, k = Config.k)
    logging.info(f"Test Loss: {average_loss:.4f}, Precision@{Config.k}: {avg_precision:.4f}, Recall@{Config.k}: {avg_recall:.4f}, NDCG@{Config.k}: {avg_ndcg:.4f}, NDCG@2: {avg_ndcg_2:.4f}, Hit Rate@{Config.k}: {avg_hit_rate:.4f}, AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")

if __name__ == "__main__":
    main()
