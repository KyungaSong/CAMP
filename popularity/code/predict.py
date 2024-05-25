import random
import pickle
import logging
import os
import gc
from tqdm.auto import tqdm
from datetime import datetime
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets
from Model import PopPredict

########################################################### config
random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.7,
                    help="parameter for balance of pop_history and time")
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.01,
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
    processed_path = f'../../dataset/preprocessed/pop/{dataset_name}/'

    combined_df = None      
    with open(f'{processed_path}/train_df_pop.pkl', 'rb') as file:
        train_df = pickle.load(file)
    with open(f'{processed_path}/valid_df_pop.pkl', 'rb') as file:
        valid_df = pickle.load(file)
    with open(f'{processed_path}/test_df_pop.pkl', 'rb') as file:
        test_df = pickle.load(file)

    combined_df = pd.concat([train_df, valid_df, test_df])
    num_items = combined_df['item_encoded'].nunique()
    num_cats = combined_df['cat_encoded'].nunique()
    num_stores = combined_df['store_encoded'].nunique()
    max_time = combined_df["unit_time"].max()
    
    if combined_df is not None:
        del combined_df
    if 'df' in locals():
        del df
    if 'df_meta' in locals():
        del df_meta
    gc.collect()

    return train_df, valid_df, test_df, num_items, num_cats, num_stores, max_time

def load_model_state(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k in model_state_dict and model_state_dict[k].shape == v.shape:
            new_state_dict[k] = v
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

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
    setup_logging()
    dataset_name = config.dataset
    train_df, valid_df, test_df, num_items, num_cats, num_stores, max_time = load_data(dataset_name)

    gc.collect()

    combined_df = pd.concat([train_df, valid_df, test_df])

    test_dataset = create_datasets(combined_df, combined_df, combined_df)[2]
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PopPredict(False, config, num_items, num_cats, num_stores, max_time).to(device)

    # date_str = datetime.now().strftime('%Y%m%d')
    date_str = '0523'
    best_epoch = '19'
    latest_checkpoint = f'../../model/pop/{date_str}/{dataset_name}_checkpoint_epoch_{best_epoch}.pt'
    if os.path.exists(latest_checkpoint):
        load_model_state(model, latest_checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint {latest_checkpoint} not found")

    outputs = generate_outputs(model, test_loader, device)
    outputs_df = pd.DataFrame(outputs)

    results_df = pd.concat([combined_df.reset_index(drop=True), outputs_df], axis=1)
    results_df['conformity'] = results_df['weighted_pop_history_output'].apply(lambda x: x[0]) + results_df['weighted_time_output'].apply(lambda x: x[0])
    results_df['quality'] = results_df['weighted_sideinfo_output'].apply(lambda x: x[0]) 
    results_df = results_df[['item_id', 'timestamp', 'conformity', 'quality']]

    result_path = f'../../dataset/{dataset_name}/pop_{dataset_name}.pkl'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results_df.to_pickle(result_path)
    print(f"Results saved to {result_path}")

if __name__ == "__main__":
    main()
