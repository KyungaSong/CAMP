import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import gzip

def load_dataset(file_path):   
    df = pd.read_csv(file_path)
    df = df.rename(columns={'parent_asin': 'item_id'})          
    return df

def get_history(group):
    history = []
    histories = []
    for item in group:
        histories.append(list(history))
        history.append(item)
    return histories

def calculate_ranges(group, k_m, k_s):
    group['mid_len'] = group['timestamp'].apply(lambda x: ((group['timestamp'] >= x - k_m) & (group['timestamp'] < x)).sum())
    group['short_len'] = group['timestamp'].apply(lambda x: ((group['timestamp'] >= x - k_s) & (group['timestamp'] < x)).sum())
    return group[['mid_len', 'short_len']] 

def encode_history_items(item_mapping, max_length, history):
    if not history:
        return [0] * max_length
    encoded_items = [item_mapping[item] for item in history]
    return [0] * (max_length - len(encoded_items)) + encoded_items

def generate_negative_samples(all_item_ids, positive_item_id, history, num_samples=4):
    non_interacted_items = list(all_item_ids - set(history) - {positive_item_id})
    neg_samples = random.sample(non_interacted_items, min(num_samples, len(non_interacted_items)))
    return neg_samples

def can_items(all_item_ids, row): 
    history = row['history_encoded']
    result = list(all_item_ids - set(history))
    return result

def preprocess_df(df, user_encoder, item_encoder, k_m, k_s, max_length):  
    df['history'] = df.groupby('user_id')['item_id'].transform(get_history)  
    ranges_df = df.groupby('user_id', group_keys=False).apply(lambda x: calculate_ranges(x, k_m, k_s), include_groups=False)
    df = pd.concat([df, ranges_df], axis=1) 

    df['user_encoded'] = user_encoder.transform(df['user_id'])
    df['item_encoded'] = item_encoder.transform(df['item_id']) + 1

    item_mapping = dict(zip(item_encoder.classes_, range(1, len(item_encoder.classes_)+1)))    
    df['history_encoded'] = df['history'].apply(lambda x: encode_history_items(item_mapping, max_length, x))

    all_item_ids = set(range(1, len(item_encoder.classes_)+1))   

    # Splitting the DataFrame into train, validation, and test sets
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    train_df = df.groupby('user_id').apply(lambda x: x.iloc[:-2], include_groups=False).reset_index(drop=True)
    valid_df = df.groupby('user_id').apply(lambda x: x.iloc[-2:-1], include_groups=False).reset_index(drop=True)
    test_df = df.groupby('user_id').apply(lambda x: x.iloc[-1:], include_groups=False).reset_index(drop=True)       
         
    # Generating negative samples for validation and test datasets
    train_df['neg_items'] = train_df.apply(lambda x: generate_negative_samples(all_item_ids, x['item_encoded'], x['history_encoded'], num_samples=4), axis=1)
    valid_df['neg_items'] = valid_df.apply(lambda x: generate_negative_samples(all_item_ids, x['item_encoded'], x['history_encoded'], num_samples=4), axis=1)
    test_df['neg_items'] = test_df.apply(lambda x: generate_negative_samples(all_item_ids, None, x['history_encoded'], num_samples=49), axis=1)
    
    return train_df, valid_df, test_df

class MakeDataset(Dataset):
    def __init__(self, users, items, histories, mid_lens, short_lens, neg_items=None):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.histories = [torch.tensor(h, dtype=torch.long) for h in histories]
        self.mid_lens = torch.tensor(mid_lens, dtype=torch.int)
        self.short_lens = torch.tensor(short_lens, dtype=torch.int)
        # Only initialize neg_items if provided
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
            'history': self.histories[idx],
            'mid_len': self.mid_lens[idx],
            'short_len': self.short_lens[idx]
        }
        if self.neg_items is not None:
            data['neg_items'] = self.neg_items[idx]
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['user_encoded'], train_df['item_encoded'], train_df['history_encoded'],
        train_df['mid_len'], train_df['short_len'], train_df['neg_items']
    )
    valid_dataset = MakeDataset(
        valid_df['user_encoded'], valid_df['item_encoded'], valid_df['history_encoded'],
        valid_df['mid_len'], valid_df['short_len'], valid_df['neg_items']
    )
    test_dataset = MakeDataset(
        test_df['user_encoded'], test_df['item_encoded'], test_df['history_encoded'],
        test_df['mid_len'], test_df['short_len'], test_df['neg_items']
    )
    return train_dataset, valid_dataset, test_dataset