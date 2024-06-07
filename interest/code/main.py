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
parser.add_argument("--batch_size", type=int, default=128,
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

parser.add_argument("--dataset", type=str, default='Sports_and_Outdoors',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", action="store_true",
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument('--no_mid', action="store_true", 
                    help='flag to indicate if model has mid-term module')
parser.add_argument('--no_con', action="store_true", 
                    help='flag to indicate if model has conformity module')
parser.add_argument('--no_qlt', action="store_true", 
                    help='flag to indicate if model has quality module')

parser.add_argument('--discrepancy_loss_weight', type=float, default=0.01, 
                    help='Loss weight for discrepancy between long and short term user embedding.')
parser.add_argument('--regularization_weight', type=float, default=0.0001, 
                    help='weight for L2 regularization applied to model parameters')


args = parser.parse_args()
config = Config(args=args)

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
    pop_file_path = f'{dataset_path}pop_{dataset_name}.pkl'
    processed_path = f'{dataset_path}preprocessed/'

    if os.path.exists(f'{processed_path}/train_df.pkl') and os.path.exists(f'{processed_path}/valid_df.pkl') and os.path.exists(f'{processed_path}/test_df.pkl') and config.data_preprocessed:
        with open(f'{processed_path}/train_df.pkl', 'rb') as file:
            train_df = pickle.load(file)
        with open(f'{processed_path}/valid_df.pkl', 'rb') as file:
            valid_df = pickle.load(file)
        with open(f'{processed_path}/test_df.pkl', 'rb') as file:
            test_df = pickle.load(file)
        
        combined_df = pd.concat([train_df, valid_df, test_df])
        num_users = combined_df['user_encoded'].max() + 1
        num_items = combined_df['item_encoded'].max() + 1
        num_cats = combined_df['cat_encoded'].max() + 1

        print("Processed files already exist. Skipping dataset preparation.")
        print(f'df: {len(combined_df)}, num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}')

    else:
        try:
            df = load_dataset(review_file_path)
            df_pop = load_dataset(pop_file_path)

            num_users = df['user_encoded'].max() + 1
            num_items = df['item_encoded'].max() + 1
            num_cats = df['cat_encoded'].max() + 1
            print(f'df: {len(df)}, num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}')

            train_df, valid_df, test_df = preprocess_df(df, df_pop, config)
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
            date_str = datetime.now().strftime('%Y%m%d')
            train_df.to_pickle(f'{processed_path}/train_df_{date_str}.pkl')
            valid_df.to_pickle(f'{processed_path}/valid_df_{date_str}.pkl')
            test_df.to_pickle(f'{processed_path}/test_df_{date_str}.pkl')
        except Exception as e:
            logging.error(f"Error during data preparation: {str(e)}")
            raise
    
    return train_df, valid_df, test_df, num_users, num_items, num_cats

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    model_dataset_path = f'{config.model_path}{config.dataset}/'
    if not os.path.exists(model_dataset_path):
            os.makedirs(model_dataset_path)

    setup_logging(config.dataset)
        
    print(f"Data preprocessing for dataset {config.dataset}......")
    train_df, valid_df, test_df, num_users, num_items, num_cats = load_data(config.dataset)

    print("Create datasets......")
    train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)

    print("Making Data loader......")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    # device = torch.device(f"cuda:{rank}" if use_cuda else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = DDP(model, device_ids=[rank]) if use_cuda else model    
    
    # learning_rates = [0.01, 0.001, 0.0001]
    # batch_sizes = [64, 128]
    # embedding_dims = [64, 128]
    learning_rates = [0.01]
    batch_sizes = [64]
    embedding_dims = [64]

    best_loss = float('inf')
    best_model_params = {}
    best_model = None

    if not config.test_only:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for embedding_dim in embedding_dims:
                    print(f"{config.dataset}_no_mid_{config.no_mid}_with lr={lr}, batch_size={batch_size}, embedding_dim={embedding_dim}")

                    config.lr = lr
                    config.batch_size = batch_size
                    config.embedding_dim = embedding_dim                      

                    model = CAMP(num_users, num_items, num_cats, config).to(device)
                    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
                    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
                    early_stopping = EarlyStopping(patience=10, verbose=True)

                    for epoch in range(config.num_epochs):
                        train_loss = train(model, train_loader, optimizer, device)                            
                        valid_loss = evaluate(model, valid_loader, device)
                        scheduler.step()

                        logging.info(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
                        if valid_loss < best_loss:
                            best_loss = valid_loss
                            best_model_params = {'lr': config.lr, 'batch_size': config.batch_size, 'embedding_dim': config.embedding_dim, 'epoch': epoch}
                            best_model = model.state_dict()

                        early_stopping(valid_loss)
                        if early_stopping.early_stop:
                            print("Early stopping triggered")
                            break

        if best_model is not None:
            print(f"Best Model Parameters: {best_model_params}")
            date_str = datetime.now().strftime('%y%m%d')
            config.embedding_dim = best_model_params['embedding_dim']
            model = CAMP(num_users, num_items, num_cats, config).to(device)
            model.load_state_dict(best_model)
            model_save_path = f'../../model/{config.dataset}/'
            final_save_path = f'{model_save_path}{date_str}_best_model.pt'
            if not os.path.exists(os.path.dirname(model_save_path)):
                os.makedirs(os.path.dirname(model_save_path))
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_dim': best_model_params['embedding_dim'],  # Save the best embedding dimension
                'lr': best_model_params['lr'],
                'batch_size': best_model_params['batch_size']
            }, final_save_path)
    else:
        date_str = input("Enter the date string of the saved model (format: yymmdd): ")
        model_path = f'../../model/{config.dataset}/{date_str}_best_model.pt'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            config.embedding_dim = checkpoint['embedding_dim']
            model = CAMP(num_users, num_items, num_cats, config).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No model found at {model_path}")
        
    average_loss, avg_precision, avg_recall, avg_ndcg, avg_hit_rate, avg_auc, avg_mrr = test(model, test_loader, device, k=config.k)
    logging.info(f"Best Model --- Test Loss: {average_loss:.4f}, Pre@{config.k}: {avg_precision:.4f}, Rec@{config.k}: {avg_recall:.4f}, NDCG@{config.k}: {avg_ndcg:.4f}, HR@{config.k}: {avg_hit_rate:.4f}, AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")

if __name__ == "__main__":
    main()