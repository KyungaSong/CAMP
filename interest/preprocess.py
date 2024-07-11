import os
import random
import pandas as pd
import numpy as np
import pickle
import gc
import ast
from dateutil.relativedelta import relativedelta

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

tqdm.pandas()

def load_file(file_path):
    with open(file_path, 'rb') as file:
        result = pickle.load(file)                
    return result

def load_txt(file_path, max_length=128):
    column_names = [
        'label', 'user_encoded', 'item_encoded', 'cat_encoded', 
        'conformity', 'quality', 'unit_time',
        'item_his_encoded', 'cat_his_encoded', 'con_his', 'qlt_his'
    ]
    df = pd.read_csv(file_path, delimiter='\t', names=column_names)

    type_conversion = {
        'label': int,
        'user_encoded': int,
        'item_encoded': int,
        'cat_encoded': int,
        'conformity': float,
        'quality': float,
        'unit_time': int,
    }

    for col, dtype in type_conversion.items():
        df[col] = df[col].astype(dtype)

    int_columns = ['item_his_encoded', 'cat_his_encoded']
    float_columns = ['con_his', 'qlt_his']

    def pad_or_truncate(item_list, max_length):
        item_list = item_list[-max_length:] if len(item_list) > max_length else [0] * (max_length - len(item_list)) + item_list
        return item_list

    for col in int_columns:
        df[col] = df[col].apply(lambda x: pad_or_truncate(list(map(int, x.split(','))), max_length))
    for col in float_columns:
        df[col] = df[col].apply(lambda x: pad_or_truncate(list(map(float, x.split(','))), max_length))

    return df

def split_data(df, data_type, split_path):

    num_users = df['user_encoded'].max() + 1
    max_time = df['unit_time'].max()
    print('max_time', max_time)

    if data_type == 'reg':
        temp_df, test_df = train_test_split(df, test_size=0.14, random_state=42)
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif data_type == 'unif':
        test_size = int(len(df) * 0.2)
        max_sample_size = test_size // df['item_encoded'].nunique()
        test_df = df.groupby('item_encoded').apply(lambda x: x.sample(n=min(len(x), max_sample_size), random_state=42)).reset_index(level=0, drop=True)
        temp_df = df.drop(test_df.index)  # MultiIndex를 제거한 후의 인덱스를 사용
        temp_df = temp_df.reset_index(drop=True) 
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif data_type == 'seq':
        train_df = df[df['unit_time'] < max_time - 1]
        rest_data = df[df['unit_time'] >= max_time - 1].reset_index(drop=True)
        val_user_set = np.random.choice(np.arange(num_users), int(num_users / 2), replace=False)
        valid_df = rest_data[rest_data['user_encoded'].isin(val_user_set)].reset_index(drop=True)
        test_df = rest_data[~rest_data['user_encoded'].isin(val_user_set)].reset_index(drop=True)
    else:
        raise ValueError("Invalid data_type. Please enter 'reg', 'unif', or 'seq'.")

    total_length = len(df)
    train_ratio = len(train_df) / total_length
    valid_ratio = len(valid_df) / total_length
    test_ratio = len(test_df) / total_length

    print(f"Train ratio: {train_ratio:.2f}, Valid ratio: {valid_ratio:.2f}, Test ratio: {test_ratio:.2f}")

    with open(split_path, 'w') as file:
        # split, user, item, category, timestamp, unit_time
        for _, row in train_df.iterrows():
            file.write(f"train\t{row['user_encoded']}\t{row['item_encoded']}\t{row['cat_encoded']}\t{row['timestamp']}\t{row['unit_time']}\n")
        for _, row in valid_df.iterrows():
            file.write(f"valid\t{row['user_encoded']}\t{row['item_encoded']}\t{row['cat_encoded']}\t{row['timestamp']}\t{row['unit_time']}\n")
        for _, row in test_df.iterrows():
            file.write(f"test\t{row['user_encoded']}\t{row['item_encoded']}\t{row['cat_encoded']}\t{row['timestamp']}\t{row['unit_time']}\n")

    return

def save_pos_sample(split_path, pop_dict, pos_train_path, pos_valid_path, pos_test_path):    
    
    df = pd.read_csv(split_path, sep="\t", header=None, names=['split', 'user', 'item', 'category', 'timestamp', 'unit_time'])
    df = df.sort_values(by=['user', 'timestamp']).reset_index(drop=True)

    f_train = open(pos_train_path, "w")
    f_valid = open(pos_valid_path, "w")
    f_test = open(pos_test_path, "w")
    
    current_user = None
    item_history = ""
    cat_history = ""
    conformity_history = ""
    quality_history = ""
    
    for _, row in df.iterrows():
        split = row['split']
        item = row['item']
        unit_time = row['unit_time']
        key = (item, unit_time)
        if key in pop_dict:
            pop_data = pop_dict[key]
            conformity = pop_data['conformity']
            quality = pop_data['quality']
        if row['user'] != current_user:
            current_user = row['user']
            item_history = ""
            cat_history = ""
            conformity_history = ""
            quality_history = ""
        
        item_history += f"{row['item']},"
        cat_history += f"{row['category']},"
        conformity_history += f"{conformity},"
        quality_history += f"{quality},"

        if split == 'train':
            file = f_train
        elif split == 'valid':
            file = f_valid
        elif split == 'test':
            file = f_test
        
        # label user_encoded item_encoded cat_encoded conformity quality unit_time item_his_encoded cat_his_encoded con_his qlt_his
        file.write(f"1\t{current_user}\t{item}\t{row['category']}\t{conformity}\t{quality}\t{row['unit_time']}\t{item_history.rstrip(',')}\t{cat_history.rstrip(',')}\t{conformity_history.rstrip(',')}\t{quality_history.rstrip(',')}\n")
    
    f_train.close()
    f_valid.close()
    f_test.close()

def neg_samples_for_row(all_item_ids, item_encoded, item_his_encoded_set, num_samples, item_to_cat, pop_dict, unit_time):
    candidate_items = list(all_item_ids - item_his_encoded_set - {item_encoded})
    random.shuffle(candidate_items)

    neg_samples = []
    valid_sample_count = 0
    for item in candidate_items:
        if valid_sample_count >= num_samples:
            break

        key = (item, unit_time)
        if key in pop_dict:
            pop_data = pop_dict[key]
            conformity = pop_data['conformity']
            quality = pop_data['quality']
            cat_encoded = item_to_cat.get(item, 0)
            neg_samples.append({
                'item_encoded': item,
                'cat_encoded': int(cat_encoded),
                'conformity': conformity,
                'quality': quality                
            })
            valid_sample_count += 1
    
    return neg_samples

def save_neg_samples(input_file, output_file, num_samples, all_item_ids, item_to_cat, pop_dict):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    results = []

    for line in tqdm(lines, desc="Generating negative samples"):
        # positive sample 저장
        results.append(line.strip())

        parts = line.strip().split('\t')
        item_encoded = int(parts[2])
        unit_time = int(parts[6])
        item_his_encoded = set(map(int, parts[7].split(',')))

        # negative sample 생성
        neg_samples = neg_samples_for_row(all_item_ids, item_encoded, item_his_encoded, num_samples, item_to_cat, pop_dict, unit_time)
        
        for neg_sample in neg_samples:
            neg_sample_line = '\t'.join([
                '0',  # label
                parts[1],  # user_encoded
                str(neg_sample['item_encoded']),
                str(neg_sample['cat_encoded']),
                str(neg_sample['conformity']),
                str(neg_sample['quality']),
                parts[6],  # unit_time
                parts[7],  # item_his_encoded
                parts[8],  # cat_his_encoded
                parts[9],  # con_his
                parts[10]   # qlt_his
            ])
            results.append(neg_sample_line)

    with open(output_file, 'w') as out_file:
        out_file.write('\n'.join(results) + '\n')

def create_pop_dict(df_pop):
    pop_dict = {}
    for _, row in df_pop.iterrows():
        key = (row['item_encoded'], row['unit_time'])
        pop_dict[key] = {'conformity': row['conformity'], 'quality': row['quality']}
    return pop_dict

def preprocess_df(config):
    if not os.path.exists(config.processed_path):
        os.makedirs(config.processed_path)

    df = load_file(config.review_file_path)
    df_pop = load_file(config.pop_file_path)
    df = df.copy()

    num_users = df['user_encoded'].max() + 1
    num_items = df['item_encoded'].max() + 1
    num_cats = df['cat_encoded'].max() + 1 

    df = df.sort_values(by=['user_encoded', 'timestamp'])
    item_to_cat = df.set_index('item_encoded')['cat_encoded'].to_dict()
    pop_dict = create_pop_dict(df_pop)    
    
    split_data(df, config.data_type, config.split_path)
    save_pos_sample(config.split_path, pop_dict, config.pos_train_path, config.pos_valid_path, config.pos_test_path)

    max_item_id = df['item_encoded'].max()
    all_item_ids = set(range(1, max_item_id + 1))    
    
    print("Train dataset -----")
    save_neg_samples(config.pos_train_path, config.train_path, config.train_num_samples, all_item_ids, item_to_cat, pop_dict)
    print("Valid dataset -----")
    save_neg_samples(config.pos_valid_path, config.valid_path, config.valid_num_samples, all_item_ids, item_to_cat, pop_dict)
    print("Test dataset -----")
    save_neg_samples(config.pos_test_path, config.test_path, config.test_num_samples, all_item_ids, item_to_cat, pop_dict)
    print("All samples saved successfully.")

    del df
    gc.collect()

    return num_users, num_items, num_cats

def load_df(config):
    if config.df_preprocessed:
        print("Processed dataframe already exist. Skipping datframe preparation.")        
    else:
        num_users, num_items, num_cats = preprocess_df(config)
        
    print("Loading txt file")
    if os.path.exists(config.train_path) and os.path.exists(config.valid_path) and os.path.exists(config.test_path):
        train_df = load_txt(config.train_path)
        valid_df = load_txt(config.valid_path)
        test_df = load_txt(config.test_path)
        
        combined_df = pd.concat([train_df, valid_df, test_df])
        num_users = combined_df['user_encoded'].max() + 1
        num_items = combined_df['item_encoded'].max() + 1
        num_cats = combined_df['cat_encoded'].max() + 1

        print(f'df: {len(combined_df)}, num_users: {num_users}, num_items: {num_items}, num_cats: {num_cats}') 
    else:
        raise FileNotFoundError("One or more file paths do not exist. You need to preprocess the data.")    
    
    return train_df, valid_df, test_df, num_users, num_items, num_cats

class LazyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        user = torch.tensor(self.df['user_encoded'].iloc[idx], dtype=torch.long)
        item = torch.tensor(self.df['item_encoded'].iloc[idx], dtype=torch.long)
        cat = torch.tensor(self.df['cat_encoded'].iloc[idx], dtype=torch.long)
        con = torch.tensor(self.df['conformity'].iloc[idx], dtype=torch.float)
        qlt = torch.tensor(self.df['quality'].iloc[idx], dtype=torch.float)
        item_his = torch.tensor(self.df['item_his_encoded'].iloc[idx], dtype=torch.long)
        cat_his = torch.tensor(self.df['cat_his_encoded'].iloc[idx], dtype=torch.long)
        con_his = torch.tensor(self.df['con_his'].iloc[idx], dtype=torch.float)
        qlt_his = torch.tensor(self.df['qlt_his'].iloc[idx], dtype=torch.float)
        label = torch.tensor(self.df['label'].iloc[idx], dtype=torch.long)
        
        data = {
            'user': user,
            'item': item,
            'cat': cat,
            'con': con,
            'qlt': qlt,
            'item_his': item_his,
            'cat_his': cat_his,
            'con_his': con_his,
            'qlt_his': qlt_his,
            'label': label
        }
        return data

def create_dataloader(train_df, valid_df, test_df, batch_size=32, num_workers=4):
    print("making train dataset")
    train_dataset = LazyDataset(train_df)
    print("making valid dataset")
    valid_dataset = LazyDataset(valid_df)
    print("making test dataset")
    test_dataset = LazyDataset(test_df)
    print("test_dataset")

    print("creating dataloaders")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("create datasets and dataloaders done!")
    return train_loader, valid_loader, test_loader