import random
import pickle
import time
import logging
import os
from datetime import datetime
import pandas as pd
import argparse

import torch
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
parser.add_argument("--num_epochs", type=int, default=30,
                    help="training epochs")
parser.add_argument("--batch_size", type=int, default=256,
                    help="batch size for training")
parser.add_argument("--dropout_rate", type=float, default=0.5,
                    help="dropout rate for model")

parser.add_argument("--embedding_dim", type=int, default=64,
                    help="embedding size for embedding vectors")
parser.add_argument("--hidden_dim", type=int, default=128,
                    help="size of the hidden layer embeddings")
parser.add_argument("--output_dim", type=int, default=1,
                    help="size of the output layer embeddings")

parser.add_argument("--time_unit", type=int, default=1000*60*60*24,
                    help="smallest time unit for model training(default: day)")
parser.add_argument("--pop_time_unit", type=int, default=30*3,
                    help="smallest time unit for item popularity statistic(default: 3 month)")
parser.add_argument("--k_m", type=int, default=3*12*30,
                    help="length of mid interest(default: 3 year)")
parser.add_argument("--k_s", type=int, default=6*30,
                    help="length of short interest(default: 6 month)")
parser.add_argument("--k", type=int, default=20,
                    help="value of k for evaluation metrics")

parser.add_argument("--dataset", type=str, default='sampled_Home_and_Kitchen',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", action="store_true",
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument('--no_mid', action="store_true", 
                    help='flag to indicate if model has mid-term module')

parser.add_argument('--discrepancy_loss_weight', type=float, default=0.01, 
                    help='Loss weight for discrepancy between long and short term user embedding.')
parser.add_argument('--regularization_weight', type=float, default=0.0001, 
                    help='weight for L2 regularization applied to model parameters')


args = parser.parse_args()
config = Config(args=args)

def setup(rank, world_size, use_cuda):
    if use_cuda:
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def setup_logging(dataset_name):
    log_dir = os.path.abspath('../../log')
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
    review_file_path = f'{dataset_path}{dataset_name}.pkl'
    meta_file_path = f'{dataset_path}meta_{dataset_name}.pkl'
    processed_path = f'../../dataset/preprocessed/{dataset_name}/'
    pop_file_path = f'../../dataset/{dataset_name}/pop_{dataset_name}.pkl'

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

        print("Processed files already exist. Skipping dataset preparation.")
        print(f'df: {len(combined_df)}, num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}')

    else:
        try:
            df = load_dataset(review_file_path, type='review')
            df_meta = load_dataset(meta_file_path, type='meta')
            df_pop = load_dataset(pop_file_path, type='pop')

            num_users = df['user_id'].nunique()
            num_items = df['item_id'].nunique()
            num_cats = df_meta['category'].nunique()
            print(f'df: {len(df)}, num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}')

            train_df, valid_df, test_df = preprocess_df(df, df_meta, df_pop, config)
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%Y%m%d')
            train_df.to_pickle(f'{processed_path}/train_df_{date_str}_{num_users}_{num_items}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_{date_str}_{num_users}_{num_items}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_{date_str}_{num_users}_{num_items}.pkl')
        except Exception as e:
            logging.error(f"Error during data preparation: {str(e)}")
            raise
    
    return train_df, valid_df, test_df, num_users, num_items, num_cats

def main(rank, world_size, use_cuda):

    model_dataset_path = f'{config.model_path}{config.dataset}/'
    if not os.path.exists(model_dataset_path):
            os.makedirs(model_dataset_path)

    setup(rank, world_size, use_cuda)
    if rank == 0:  
        setup_logging(config.dataset)
        
    print("Data preprocessing......")
    train_df, valid_df, test_df, num_users, num_items, num_cats = load_data(config.dataset)

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
    model = CAMP(num_users, num_items, num_cats, config).to(device)
    model = DDP(model, device_ids=[rank]) if use_cuda else model    
    
    if not config.test_only:
        optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        logging.info(f"{config.dataset}_no_mid_{config.no_mid}-----------------------------------------------------------------------------------")
        for epoch in range(config.num_epochs):
            train_sampler.set_epoch(epoch)
            if rank == 0:
                print(f"Rank {rank}, Epoch {epoch+1} -----------------------------------")
            train_loss = train(model, train_loader, optimizer, device, rank)
            scheduler.step()
            valid_loss = evaluate(model, valid_loader, device, rank)
            if rank == 0:                
                logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
                torch.save(model.state_dict(), f'{model_dataset_path}mid_{config.no_mid}_epoch_{epoch}.pt')
            
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    if rank == 0:
        for i in range(config.num_epochs):
            latest_checkpoint = f'{model_dataset_path}mid_{config.no_mid}_epoch_{i}.pt'
            if not os.path.exists(latest_checkpoint):
                break
            model.load_state_dict(torch.load(latest_checkpoint))
            model.eval()
            average_loss, avg_precision, avg_recall, avg_ndcg, avg_hit_rate, avg_auc, avg_mrr = test(model, test_loader, device, rank, k=config.k)
            logging.info(f"epoch_{i}---Test Loss: {average_loss:.4f}, Pre@{config.k}: {avg_precision:.6f}, Rec@{config.k}: {avg_recall:.6f}, NDCG@{config.k}: {avg_ndcg:.6f}, HR@{config.k}: {avg_hit_rate:.6f}, AUC: {avg_auc:.6f}, MRR: {avg_mrr:.6f}")

    cleanup()

if __name__ == "__main__":
    # n_gpus = torch.cuda.device_count()
    # n_gpus = 3
    n_gpus = 1
    if n_gpus > 0:
        print(f"Let's use {n_gpus} GPUs!")
        spawn(main, args=(n_gpus, True), nprocs=n_gpus)
    else:
        main(rank=0, world_size=1, use_cuda=False)
    