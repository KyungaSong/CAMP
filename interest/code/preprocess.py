import random
import pandas as pd
import numpy as np
import pickle
from functools import lru_cache
import torch
from torch.utils.data import Dataset

from tqdm.auto import tqdm

def load_dataset(file_path, meta = False):
    with open(file_path, 'rb') as file:
        df = pickle.load(file) 
    if meta:
        necessary_columns = ['average_rating', 'rating_number', 'store', 'parent_asin', 'categories']
        df = df[necessary_columns].rename(columns={'average_rating': 'avg_rating', 'parent_asin': 'item_id'})
        df['category'] = df['categories'].apply(lambda x: x[1] if len(x) > 1 else (x[0] if len(x) == 1 else None))
    else:        
        df = df.rename(columns={'parent_asin': 'item_id'})          
    return df

def encode_column(column, pad = False):
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

def generate_negative_samples(all_item_ids, positive_item_id, history):
    non_interacted_items = list(all_item_ids - set(history) - {positive_item_id})
    return non_interacted_items

@lru_cache(maxsize=1024)  
def generate_negative_samples_cached(all_item_ids, positive_item_id, history_tuple, num_samples=4):
    history = set(history_tuple)  
    non_interacted_items = list(all_item_ids - history - {positive_item_id})
    neg_samples = random.sample(non_interacted_items, min(num_samples, len(non_interacted_items)))
    return neg_samples

def preprocess_df(df, Config):     
    df['user_encoded'] = encode_column(df['user_id'])
    df['item_encoded'] = encode_column(df['item_id'], pad = True)
    df['cat_encoded'] = encode_column(df['category'], pad = True)
    item_to_cat_dict = dict(zip(df['item_encoded'], df['cat_encoded']))
    print("df\n", df)
    max_item_id = df['item_encoded'].max()
    all_item_ids = set(range(1, max_item_id + 1))

    min_time_all = df['timestamp'].min()
    df['unit_time'] = (df['timestamp'] - min_time_all) // Config.time_range

    df['item_his_encoded'] = df.groupby('user_id')['item_encoded'].transform(get_history) 
    df['cat_his_encoded'] = df.groupby('user_id')['cat_encoded'].transform(get_history) 

    print('calculate ranges start')
    ranges_df = df.groupby('user_id', group_keys=False).apply(lambda x: calculate_ranges(x, Config.k_m, Config.k_s), include_groups=False)
    df.reset_index(drop=True, inplace=True)
    ranges_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, ranges_df], axis=1) 
    print('calculate ranges end')

    df = df[['user_id', 'user_encoded', 'item_encoded', 'cat_encoded', 'item_his_encoded', 'cat_his_encoded', 'unit_time', 'mid_len', 'short_len']]
    # df['item_his_encoded_set'] = df['item_his_encoded'].apply(set)
    print("df:\n", df)
    
    print('Split into train, valid, test dataframe...')
    train_df = df.groupby('user_id').apply(lambda x: x.iloc[:-2], include_groups=False).reset_index(drop=True)
    valid_df = df.groupby('user_id').apply(lambda x: x.iloc[-2:-1], include_groups=False).reset_index(drop=True)
    test_df = df.groupby('user_id').apply(lambda x: x.iloc[-1:], include_groups=False).reset_index(drop=True)       

    print('Making negative candidates start')
    
    # train_df['can_neg'] = train_df.apply(lambda x: generate_negative_samples(all_item_ids, x['item_encoded'], x['item_his_encoded']), axis=1)
    # print('train df end')    
    # valid_df['can_neg'] = valid_df.apply(lambda x: generate_negative_samples(all_item_ids, x['item_encoded'], x['item_his_encoded']), axis=1)
    # print('train df end') 
    # test_df['can_neg'] = test_df.apply(lambda x: generate_negative_samples(all_item_ids, None, x['item_his_encoded']), axis=1)
    # print('test df end')
    tqdm.pandas(desc="Generating negative samples for training dataset")
    train_df['neg_items'] = train_df.progress_apply(
    lambda x: generate_negative_samples_cached(
        frozenset(all_item_ids), 
        x['item_encoded'], 
        tuple(x['item_his_encoded']), 
        num_samples= Config.train_num_samples
    ), axis=1)
    tqdm.pandas(desc="Generating negative samples for valid dataset")
    valid_df['neg_items'] = valid_df.progress_apply(lambda x: generate_negative_samples_cached(
        frozenset(all_item_ids), 
        x['item_encoded'], 
        tuple(x['item_his_encoded']), 
        num_samples = Config.valid_num_samples
    ), axis=1)
    tqdm.pandas(desc="Generating negative samples for test dataset")
    test_df['neg_items'] = test_df.progress_apply(lambda x: generate_negative_samples_cached(
        frozenset(all_item_ids), 
        x['item_encoded'], 
        tuple(x['item_his_encoded']), 
        num_samples = Config.test_num_samples
    ), axis=1)
    test_df['neg_items'] = test_df['neg_items'] + test_df['item_encoded'].apply(lambda x: [x])
    print('Making negative candidates end')
    return train_df, valid_df, test_df, item_to_cat_dict

class MakeDataset(Dataset):
    def __init__(self, users, items, cats, item_histories, cat_histories, mid_lens, short_lens, neg_items=None):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)  
        self.cats = torch.tensor(cats, dtype=torch.long)       
        self.item_histories = [torch.tensor(h, dtype=torch.long) for h in item_histories]
        self.cat_histories = [torch.tensor(c, dtype=torch.long) for c in cat_histories]
        self.mid_lens = torch.tensor(mid_lens, dtype=torch.int)
        self.short_lens = torch.tensor(short_lens, dtype=torch.int)
        # Only initialize can_neg if provided
        # if can_neg is not None:
        #     self.can_neg = [torch.tensor(n, dtype=torch.long) for n in can_neg]
        # else:
        #     self.can_neg = None
        if neg_items is not None:
            self.neg_items = [torch.tensor(n, dtype=torch.long) for n in neg_items]
        else:
            self.neg_items = None

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        data = {
            'user': self.users[idx],
            'item': self.items[idx],   
            'cat': self.cats[idx],           
            'item_his': self.item_histories[idx],
            'cat_his': self.cat_histories[idx],
            'mid_len': self.mid_lens[idx],
            'short_len': self.short_lens[idx]
        }
        # if self.can_neg is not None:
        #     data['can_neg'] = self.can_neg[idx]
        # return data
        if self.neg_items is not None:
            data['neg_items'] = self.neg_items[idx]
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['user_encoded'], train_df['item_encoded'], train_df['cat_encoded'], train_df['item_his_encoded'], train_df['cat_his_encoded'],
        train_df['mid_len'], train_df['short_len'], train_df['neg_items']
    )
    valid_dataset = MakeDataset(
        valid_df['user_encoded'], valid_df['item_encoded'], valid_df['cat_encoded'], valid_df['item_his_encoded'], valid_df['cat_his_encoded'],
        valid_df['mid_len'], valid_df['short_len'], valid_df['neg_items']
    )
    test_dataset = MakeDataset(
        test_df['user_encoded'], test_df['item_encoded'], test_df['cat_encoded'], test_df['item_his_encoded'], test_df['cat_his_encoded'],
        test_df['mid_len'], test_df['short_len'], test_df['neg_items']
    )
    return train_dataset, valid_dataset, test_dataset