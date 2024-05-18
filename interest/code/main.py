import random
import pickle
import time
import logging
import os
from datetime import datetime
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn
from torch.utils.data.distributed import DistributedSampler

from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets
from Model import CAMP
from training_utils import train, evaluate, test, EarlyStopping

random.seed(2024) 
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="training epochs")
parser.add_argument("--batch_size", type=int, default=256,
                    help="batch size for training")
parser.add_argument("--embedding_dim", type=int, default=128,
                    help="embedding size for embedding vectors")
parser.add_argument("--hidden_dim", type=int, default=256,
                    help="size of the hidden layer embeddings")
parser.add_argument("--output_dim", type=int, default=1,
                    help="size of the output layer embeddings")

parser.add_argument("--k_m", type=int, default=3*12,
                    help="length of mid interest")
parser.add_argument("--k_s", type=int, default=6,
                    help="length of short interest")
parser.add_argument("--k", type=int, default=20,
                    help="value of k for evaluation metrics")

parser.add_argument("--dataset", type=str, default='sampled_Home_and_Kitchen',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", type=bool, default=False,
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", type=bool, default=False,
                    help="flag to indicate if only testing should be performed")

args = parser.parse_args()
config = Config(args=args)

def setup(rank, world_size, use_cuda):
    if use_cuda:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def setup_logging():
    logging.basicConfig(filename='../../log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(dataset_name):    
    dataset_path = f'../../dataset/{dataset_name}/'
    review_file_path = f'{dataset_path}{dataset_name}.pkl'
    meta_file_path = f'{dataset_path}meta_{dataset_name}.pkl'
    processed_path = f'../../dataset/preprocessed/{dataset_name}/'

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

def main(rank, world_size, use_cuda):
    setup(rank, world_size, use_cuda)
    if rank == 0:  
        setup_logging()
    print("Data preprocessing......")
    dataset_name = 'sampled_Home_and_Kitchen'
    train_df, valid_df, test_df, num_users, num_items, num_cats, item_to_cat_dict = load_data(dataset_name = dataset_name)

    print("Create datasets......")
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
    print("Making Data loader......")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler)

    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    model = CAMP(num_users, num_items, num_cats, Config).to(device)
    model = DDP(model, device_ids=[rank]) if use_cuda else model    

    if not config.test_only:
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        for epoch in range(config.num_epochs):
            train_sampler.set_epoch(epoch)
            print(f"Rank {rank}, Epoch {epoch} -------")
            train_loss = train(model, train_loader, optimizer, item_to_cat_dict, device)
            scheduler.step()
            valid_loss, valid_accuracy = evaluate(model, valid_loader, item_to_cat_dict, device)
            if rank == 0:
                logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Valid Acc: {valid_accuracy}')
                torch.save(model.state_dict(), f'../../model/{dataset_name}_checkpoint_epoch_{epoch}.pt')
            
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    if rank == 0 and use_cuda:
        for i in range(config.num_epochs):
            latest_checkpoint = f'../../model/checkpoint_epoch_{i}.pt'
            model.load_state_dict(torch.load(latest_checkpoint))
            model.eval()
            average_loss, all_top_k_items, avg_precision, avg_recall, avg_ndcg, avg_ndcg_2, avg_hit_rate, avg_auc, avg_mrr = test(model, test_loader, item_to_cat_dict, device, k=config.k)
            logging.info(f"{dataset_name}_epoch_{i} ------\n Test Loss: {average_loss:.4f}, Precision@{config.k}: {avg_precision:.4f}, Recall@{config.k}: {avg_recall:.4f}, NDCG@{config.k}: {avg_ndcg:.4f}, NDCG@2: {avg_ndcg_2:.4f}, Hit Rate@{config.k}: {avg_hit_rate:.4f}, AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")

    cleanup()

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        print(f"Let's use {n_gpus} GPUs!")
        spawn(main, args=(n_gpus, True), nprocs=n_gpus)
    else:
        main(rank=0, world_size=1, use_cuda=False)
    