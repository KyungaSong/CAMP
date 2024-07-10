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
from torch.utils.data import DataLoader

from config import Config
from preprocess import create_datasets
from Model import PopPredict

########################################################### config
random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument("--alpha", type=float, default=0.7,
                    help="parameter for balance of pop_history and time")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size for training")
parser.add_argument("--lr", type=float, default=0.01,
                    help="learning rate")
parser.add_argument("--num_epochs", type=int, default=50,
                    help="training epochs")
parser.add_argument("--time_unit", type=int, default=1000*60*60*24,
                    help="smallest time unit for model training")
parser.add_argument("--pop_time_unit", type=int, default=3*30,
                    help="smallest time unit for item popularity statistic")
parser.add_argument("--dataset", type=str, default='Home_and_Kitchen',
                    help="dataset file name")
parser.add_argument("--data_preprocessed", action="store_true",
                    help="flag to indicate if the input data has already been preprocessed")
parser.add_argument("--test_only", action="store_true",
                    help="flag to indicate if only testing should be performed")
parser.add_argument("--embedding_dim", type=int, default=128,
                    help="embedding size for embedding vectors")
parser.add_argument("--wt_pop", type=float, default=1.0,
                    help="Weight parameter for balancing the loss contribution of pop_history")
parser.add_argument("--wt_time", type=float, default=1.0,
                    help="Weight parameter for balancing the loss contribution of release_time")
parser.add_argument("--wt_side", type=float, default=2.0,
                    help="Weight parameter for balancing the loss contribution of side_information")

args = parser.parse_args()
config = Config(args=args)

def expand_time(row, max_time):
    unit_times = range(row['release_time'], max_time + 1)
    return pd.DataFrame({
        'item_encoded': [row['item_encoded']] * len(unit_times),
        'unit_time': list(unit_times),
        'release_time': [row['release_time']] * len(unit_times),
        'pop_history': [row['pop_history']] * len(unit_times),
        'average_rating': [row['average_rating']] * len(unit_times),
        'cat_encoded': [row['cat_encoded']] * len(unit_times),
        'store_encoded': [row['store_encoded']] * len(unit_times)
    })

def load_data(dataset_name):
    processed_path = f'../../dataset/{dataset_name}/preprocessed/'
    result_file = f'{processed_path}result_df_pop.pkl'

    if os.path.exists(result_file) and config.data_preprocessed:
        with open(result_file, 'rb') as file:
            result_df = pickle.load(file)

        # Extract necessary information from the result_df
        num_items = result_df['item_encoded'].max() + 1
        num_cats = result_df['cat_encoded'].max() + 1
        num_stores = result_df['store_encoded'].max() + 1
        max_time = result_df["unit_time"].max()

        return result_df, num_items, num_cats, num_stores, max_time

    combined_df = None      
    with open(f'{processed_path}train_df_pop.pkl', 'rb') as file:
        train_df = pickle.load(file)
    with open(f'{processed_path}valid_df_pop.pkl', 'rb') as file:
        valid_df = pickle.load(file)
    with open(f'{processed_path}test_df_pop.pkl', 'rb') as file:
        test_df = pickle.load(file)

    combined_df = pd.concat([train_df, valid_df, test_df])
    max_time = combined_df["unit_time"].max()

    first_df = combined_df.drop_duplicates(subset='item_encoded', keep='first')
    first_df = first_df[['item_encoded', 'release_time', 'pop_history','average_rating', 'cat_encoded', 'store_encoded']]
    num_items = first_df['item_encoded'].max() + 1
    num_cats = first_df['cat_encoded'].max() + 1
    num_stores = first_df['store_encoded'].max() + 1

    result_df = pd.concat([expand_time(row, max_time) for _, row in first_df.iterrows()]).reset_index(drop=True)

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    result_df.to_pickle(result_file)

    if 'df' in locals():
        del df
    gc.collect()

    return result_df, num_items, num_cats, num_stores, max_time

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
    dataset_name = config.dataset
    combined_df, num_items, num_cats, num_stores, max_time = load_data(dataset_name)

    gc.collect()

    test_dataset = create_datasets(combined_df, combined_df, combined_df)[2]
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PopPredict(config, num_items, num_cats, num_stores, max_time).to(device)

    latest_checkpoint = f'../../model/pop/{dataset_name}/best_model.pt'

    if os.path.exists(latest_checkpoint):
        load_model_state(model, latest_checkpoint)
    else:
        raise FileNotFoundError(f"Checkpoint {latest_checkpoint} not found")

    outputs = generate_outputs(model, test_loader, device)
    outputs_df = pd.DataFrame(outputs)

    results_df = pd.concat([combined_df.reset_index(drop=True), outputs_df], axis=1)
    
    results_df['time_output'] = results_df['weighted_time_output'].apply(lambda x: x[0])
    results_df['conformity'] = results_df['weighted_pop_history_output'].apply(lambda x: x[0]) + results_df['weighted_time_output'].apply(lambda x: x[0])
    results_df['quality'] = results_df['weighted_sideinfo_output'].apply(lambda x: x[0]) 
    results_df = results_df[['item_encoded', 'unit_time', 'weighted_pop_history_output', 'weighted_time_output', 'weighted_sideinfo_output', 'time_output', 'conformity', 'quality']]

    result_path = f'../../dataset/{dataset_name}/pop_{dataset_name}.pkl'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    results_df.to_pickle(result_path)
    print(f"Results saved to {result_path}")
    print("result_df zero ratio\n", len(results_df[results_df['time_output'] == 0])/len(results_df), len(results_df[results_df['conformity'] == 0])/len(results_df), len(results_df[results_df['quality'] == 0])/len(results_df))

if __name__ == "__main__":
    main()
