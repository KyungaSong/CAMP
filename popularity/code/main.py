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
from torch.optim import Adam
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import StepLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.multiprocessing import spawn

from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets
from Model import ModulePopHistory, PopPredict
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
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=20,
                    help="training epochs")
parser.add_argument("--time_unit", type=int, default=1000 * 60 * 60 * 24,
                    help="smallest time unit for model training")
parser.add_argument("--pop_time_unit", type=int, default=30,
                    help="smallest time unit for item popularity statistic")
parser.add_argument("--dataset", type=str, default='sampled_Home_and_Kitchen',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", type=bool, default=False,
                    help="lag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", type=bool, default=False,
                    help="flag to indicate if only testing should be performed")
parser.add_argument("--embedding_dim", type=int, default=128,
                    help="embedding size for embedding vectors")

args = parser.parse_args()
config = Config(args=args)



def setup(rank, world_size, use_cuda):
    if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def setup_logging():
    logging.basicConfig(filename='../../pop_log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(dataset_name):
    dataset_path = f'../../dataset/{dataset_name}/'
    review_file_path = f'{dataset_path}{dataset_name}.pkl'
    meta_file_path = f'{dataset_path}meta_{dataset_name}.pkl'
    processed_path = f'../../dataset/preprocessed/pop/{dataset_name}/'

    if os.path.exists(f'{processed_path}/train_df_pop.pkl') and os.path.exists(f'{processed_path}/valid_df_pop.pkl') and os.path.exists(f'{processed_path}/test_df_pop.pkl') and config.data_preprocessed:
        with open(f'{processed_path}/train_df_pop.pkl', 'rb') as file:
            train_df = pickle.load(file)
        with open(f'{processed_path}/valid_df_pop.pkl', 'rb') as file:
            valid_df = pickle.load(file)
        with open(f'{processed_path}/test_df_pop.pkl', 'rb') as file:
            test_df = pickle.load(file)

        combined_df = pd.concat([train_df, valid_df, test_df])
        num_users = combined_df['user_encoded'].nunique()
        num_items = combined_df['item_encoded'].nunique()
        num_cats = combined_df['cat_encoded'].nunique()
        num_stores = combined_df['store_encoded'].nunique()
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

        except Exception as e:
            raise

    return train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time

def main(rank, world_size, use_cuda):
    print(f"Rank {rank} Data preprocessing......")
    setup_logging()
    dataset_name = config.dataset
    train_df, valid_df, test_df, num_users, num_items, num_cats, num_stores, max_time = load_data(dataset_name)

    setup(rank, world_size, use_cuda)
    print(f"Rank {rank} Create datasets......")
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)

    print(f"Rank {rank} Making Data loader......")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, sampler=test_sampler)

    print(f"Rank {rank} Training......")
    device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")

    model = PopPredict(True, config, num_items, num_cats, num_stores, max_time).to(device)
    if use_cuda:
        model = DDP(model, device_ids=[rank])

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.0001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(patience=20, verbose=True)

    # Train and evaluate
    for epoch in range(config.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, rank)
        torch.cuda.synchronize()
        dist.barrier()
        # scheduler.step()

        valid_loss, valid_rmse = evaluate(model, valid_loader, device, rank)
        torch.cuda.synchronize()
        dist.barrier()

        if rank == 0:
            print(f"Epoch {epoch+1}>>> Average Train Loss: {train_loss:.4f}, Average Validation Loss: {valid_loss:.4f}, Average Valid RMSE: {valid_rmse:.4f}")
            logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Average Valid RMSE: {valid_rmse:.4f}')
            torch.save(model.state_dict(), f'../../model/pop/{dataset_name}_checkpoint_epoch_{epoch}.pt')

            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    if use_cuda:
        dist.barrier()

    # Test the model after training is complete
    if rank == 0:  # Only the main process should perform the test
        print("Testing......")
        average_loss, average_rmse = test(model, test_loader, device, rank)
        print(f"Average Test Loss: {average_loss:.4f}, Average Test RMSE: {average_rmse:.4f}")
        logging.info(f'Average Test Loss: {average_loss:.4f}, Average Test RMSE: {average_rmse:.4f}')

    # Ensure all processes reach this point before cleanup
    if use_cuda:
        dist.barrier()

    cleanup()

if __name__ == "__main__":
    n_gpus = 1
    if n_gpus > 0:
        print(f"Let's use {n_gpus} GPUs!")
        spawn(main, args=(n_gpus, True), nprocs=n_gpus)
    else:
        main(rank=0, world_size=1, use_cuda=False)
