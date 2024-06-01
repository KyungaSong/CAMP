import random
import pandas as pd
import numpy as np
import pickle
from functools import lru_cache
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

tqdm.pandas()

def load_dataset(file_path, type='pop'):
    with open(file_path, 'rb') as file:
        df = pickle.load(file) 
    if type == 'meta':
        necessary_columns = ['parent_asin', 'categories']
        df = df[necessary_columns].rename(columns={'parent_asin': 'item_id'})
        df['category'] = df['categories'].apply(lambda x: x[1] if len(x) > 1 else (x[0] if len(x) == 1 else 'Home & Kitchen'))
    elif type == 'review':
        df = df.rename(columns={'parent_asin': 'item_id'})                  
    return df

def encode_column(column, pad=False):
    frequencies = column.value_counts(ascending=False)
    if pad:
        mapping = pd.Series(index=frequencies.index, data=range(1, len(frequencies) + 1))
    else:
        mapping = pd.Series(index=frequencies.index, data=range(len(frequencies)))
    encoded_column = column.map(mapping).fillna(0).astype(int)
    return encoded_column

def get_history(group):
    group_array = np.array(group)
    histories = []
    for i in range(len(group_array)):
        history = group_array[max(0, i - 128 + 1):i + 1]  
        histories.append(np.pad(history, (128 - len(history), 0), mode='constant'))    
    return histories

def calculate_ranges(group, k_m, k_s):
    if not pd.api.types.is_datetime64_any_dtype(group['timestamp']):
        group['timestamp'] = pd.to_datetime(group['timestamp'], unit='ms')
    k_m_delta = pd.Timedelta(milliseconds=k_m)
    k_s_delta = pd.Timedelta(milliseconds=k_s)

    group.set_index('timestamp', inplace=True)
    group['mid_len'] = group.index.to_series().apply(lambda x: group.loc[x-k_m_delta:x].shape[0] - 1)
    group['short_len'] = group.index.to_series().apply(lambda x: group.loc[x-k_s_delta:x].shape[0] - 1)
    group.reset_index(inplace=True)

    return group[['mid_len', 'short_len']]

def generate_negative_samples_for_row(all_item_ids, item_encoded, item_his_encoded_set, num_samples, item_to_cat, df_pop, unit_time):
    candidate_items = list(all_item_ids - item_his_encoded_set - {item_encoded})
    random.shuffle(candidate_items)

    neg_samples = []
    valid_sample_count = 0
    for item in candidate_items:
        if valid_sample_count >= num_samples:
            break

        pop_row = df_pop[(df_pop['item_encoded'] == item) & (df_pop['unit_time'] == unit_time)]
        if not pop_row.empty:
            pop_row = pop_row.iloc[0]
            conformity = pop_row['conformity']
            quality = pop_row['quality']
            cat_encoded = item_to_cat.get(item, 0)
            neg_samples.append({
                'item_encoded': item,
                'cat_encoded': int(cat_encoded),
                'conformity': conformity,
                'quality': quality
            })
            valid_sample_count += 1

    if valid_sample_count < num_samples:
        neg_samples.extend([{
            'item_encoded': 0,
            'cat_encoded': 0,
            'conformity': 0.0,
            'quality': 0.0
        }] * (num_samples - valid_sample_count))
    
    return neg_samples

def repeat_column(arr, num_samples):
    repeated = np.repeat(arr, num_samples)
    return repeated

def expand_and_assign(df, neg_samples_df, col_name, num_samples):
    repeated = repeat_column(np.array(df[col_name]), num_samples)
    repeated_df = pd.DataFrame(repeated, columns=[col_name])
    neg_samples_df = pd.concat([neg_samples_df.reset_index(drop=True), repeated_df.reset_index(drop=True)], axis=1)
    return neg_samples_df

def generate_negative_samples_vectorized(df, df_pop, all_item_ids, num_samples, item_to_cat):
    positive_items = df['item_encoded'].values
    histories = df['item_his_encoded_set'].values
    unit_times = df['unit_time'].values

    neg_samples = []
    for idx in tqdm(range(len(df)), desc="Generating negative samples"):
        neg_samples.extend(generate_negative_samples_for_row(
            all_item_ids, positive_items[idx], histories[idx], num_samples, item_to_cat, df_pop, unit_times[idx]
        ))

    neg_samples_df = pd.DataFrame(neg_samples)

    if len(neg_samples) != len(df) * num_samples:
        raise ValueError("The length of the negative samples does not match the expected length.")

    indices = np.repeat(np.arange(len(df)), num_samples)

    neg_samples_df['user_encoded'] = df['user_encoded'].values[indices]    
    
    neg_samples_df = expand_and_assign(df, neg_samples_df, 'item_his_encoded', num_samples)
    neg_samples_df = expand_and_assign(df, neg_samples_df, 'cat_his_encoded', num_samples)
    neg_samples_df = expand_and_assign(df, neg_samples_df, 'con_his', num_samples)
    neg_samples_df = expand_and_assign(df, neg_samples_df, 'qlt_his', num_samples)

    neg_samples_df['item_his_encoded_set'] = df['item_his_encoded_set'].values[indices]
    neg_samples_df['unit_time'] = df['unit_time'].values[indices]
    neg_samples_df['mid_len'] = df['mid_len'].values[indices]
    neg_samples_df['short_len'] = df['short_len'].values[indices]
    neg_samples_df['label'] = 0

    neg_samples_df = neg_samples_df[neg_samples_df['item_encoded'] != 0]

    return neg_samples_df

def preprocess_df(df, df_meta, df_pop, config):
    df = df.merge(df_meta, on='item_id', how='left')
    df['user_encoded'] = encode_column(df['user_id'])
    df['item_encoded'] = encode_column(df['item_id'], pad=True)
    df['cat_encoded'] = encode_column(df['category'], pad=True).astype(int)

    item_to_cat = df.set_index('item_encoded')['cat_encoded'].to_dict()

    min_time_all = df['timestamp'].min()
    df['unit_time'] = (df['timestamp'] - min_time_all) // config.pop_time_unit
    max_time = df["unit_time"].max()
    df = df.merge(df_pop, on=['item_encoded', 'unit_time'], how='left')

    df['item_his_encoded'] = df.groupby('user_id')['item_encoded'].transform(get_history)
    df['cat_his_encoded'] = df.groupby('user_id')['cat_encoded'].transform(get_history)
    df['con_his'] = df.groupby('user_id')['conformity'].transform(get_history)
    df['qlt_his'] = df.groupby('user_id')['quality'].transform(get_history)

    nan_found = df['item_his_encoded'].apply(lambda x: any(np.isnan(x)))

    df['item_his_encoded_set'] = df['item_his_encoded'].apply(set)
    df['label'] = 1

    max_item_id = df['item_encoded'].max()
    all_item_ids = set(range(1, max_item_id + 1))

    ranges_df = df.groupby('user_id', group_keys=False).apply(lambda x: calculate_ranges(x, config.k_m, config.k_s), include_groups=False)
    df.reset_index(drop=True, inplace=True)
    ranges_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, ranges_df], axis=1)

    df = df[['user_encoded', 'item_encoded', 'cat_encoded', 'conformity', 'quality', 'item_his_encoded', 'item_his_encoded_set', 'cat_his_encoded', 'con_his', 'qlt_his', 'unit_time', 'mid_len', 'short_len', 'label']]

    num_users = df['user_encoded'].max() + 1

    train_df = df[df['unit_time'] < max_time - 7]
    rest_df = df[df['unit_time'] >= max_time - 7].reset_index(drop=True)

    rest_user_set = np.random.choice(np.arange(num_users + 1), int((num_users + 1) / 2), replace=False)
    valid_df = rest_df[rest_df['user_encoded'].isin(rest_user_set)].reset_index(drop=True)
    test_df = rest_df[~rest_df['user_encoded'].isin(rest_user_set)].reset_index(drop=True)

    total_length = len(df)
    train_ratio = len(train_df) / total_length
    valid_ratio = len(valid_df) / total_length
    test_ratio = len(test_df) / total_length

    print(f"Train ratio: {train_ratio:.2f}, Valid ratio: {valid_ratio:.2f}, Test ratio: {test_ratio:.2f}")

    print("Generating negative samples for train dataset")
    train_neg_df = generate_negative_samples_vectorized(train_df, df_pop, all_item_ids, config.train_num_samples, item_to_cat)
    print("Generating negative samples for valid dataset")
    valid_neg_df = generate_negative_samples_vectorized(valid_df, df_pop, all_item_ids, config.valid_num_samples, item_to_cat)
    print("Generating negative samples for test dataset")
    test_neg_df = generate_negative_samples_vectorized(test_df, df_pop, all_item_ids, config.test_num_samples, item_to_cat)

    train_df = pd.concat([train_df, train_neg_df], ignore_index=True)
    valid_df = pd.concat([valid_df, valid_neg_df], ignore_index=True)
    test_df = pd.concat([test_df, test_neg_df], ignore_index=True)

    return train_df, valid_df, test_df

class MakeDataset(Dataset):
    def __init__(self, users, items, cats, cons, qlts, item_histories, cat_histories, con_histories, qlt_histories, mid_lens, short_lens, labels):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)  
        self.cats = torch.tensor(cats, dtype=torch.long)       
        self.cons = torch.tensor(cons, dtype=torch.float)  
        self.qlts = torch.tensor(qlts, dtype=torch.float) 
        self.item_histories = [torch.tensor(h, dtype=torch.long) for h in item_histories]
        self.cat_histories = [torch.tensor(c, dtype=torch.long) for c in cat_histories]
        self.con_histories = [torch.tensor(c, dtype=torch.float) for c in con_histories]
        self.qlt_histories = [torch.tensor(q, dtype=torch.float) for q in qlt_histories]
        self.mid_lens = torch.tensor(mid_lens, dtype=torch.int)
        self.short_lens = torch.tensor(short_lens, dtype=torch.int)
        self.labels = torch.tensor(labels, dtype=torch.long)  

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        data = {
            'user': self.users[idx],
            'item': self.items[idx],   
            'cat': self.cats[idx],       
            'con': self.cons[idx],   
            'qlt': self.qlts[idx],      
            'item_his': self.item_histories[idx],
            'cat_his': self.cat_histories[idx],
            'con_his': self.con_histories[idx],
            'qlt_his': self.qlt_histories[idx],
            'mid_len': self.mid_lens[idx],
            'short_len': self.short_lens[idx],
            'label': self.labels[idx]
        }
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['user_encoded'], train_df['item_encoded'], train_df['cat_encoded'], train_df['conformity'], train_df['quality'],
        train_df['item_his_encoded'], train_df['cat_his_encoded'],  train_df['con_his'], train_df['qlt_his'], 
        train_df['mid_len'], train_df['short_len'], train_df['label']
    )
    valid_dataset = MakeDataset(
        valid_df['user_encoded'], valid_df['item_encoded'], valid_df['cat_encoded'], valid_df['conformity'], valid_df['quality'],
        valid_df['item_his_encoded'], valid_df['cat_his_encoded'], valid_df['con_his'], valid_df['qlt_his'], 
        valid_df['mid_len'], valid_df['short_len'], valid_df['label']
    )
    test_dataset = MakeDataset(
        test_df['user_encoded'], test_df['item_encoded'], test_df['cat_encoded'], test_df['conformity'], test_df['quality'],
        test_df['item_his_encoded'], test_df['cat_his_encoded'], test_df['con_his'], test_df['qlt_his'], 
        test_df['mid_len'], test_df['short_len'], test_df['label']
    )
    return train_dataset, valid_dataset, test_dataset
