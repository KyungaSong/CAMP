import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

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

def preprocess_df(df, config):     
    df['item_encoded'] = encode_column(df['item_id'], pad = True)
    df['cat_encoded'] = encode_column(df['category'], pad = True)
    df['store_encoded'] = encode_column(df['store'])    

    min_time_all = df['timestamp'].min()

    df['unit_time'] = (df['timestamp'] - min_time_all) // (config.pop_time_unit * config.time_unit)

    df = df.sort_values(by=['item_encoded', 'unit_time'])

    min_unit_time = df.groupby('item_encoded')['unit_time'].transform('min')
    df['release_time'] = min_unit_time

    max_time = df["unit_time"].max()
    time_range = list(range(0, max_time+1))

    count_per_group = df.groupby(['item_encoded', 'unit_time']).size().unstack(fill_value=0)
    count_per_group = count_per_group.reindex(columns=time_range, fill_value=0)

    df_pop = count_per_group.apply(lambda row: row.tolist(), axis=1)
    df_pop = df_pop.reset_index(name='pop_history')

    df = df.merge(df_pop, on='item_encoded', how='left')
    
    unique_items = df['item_encoded'].unique()
    np.random.shuffle(unique_items)

    num_items = len(unique_items)
    train_end = int(num_items * 0.8)
    valid_end = train_end + int(num_items * 0.1)

    train_items = unique_items[:train_end]
    valid_items = unique_items[train_end:valid_end]
    test_items = unique_items[valid_end:]

    train_df = df[df['item_encoded'].isin(train_items)]
    valid_df = df[df['item_encoded'].isin(valid_items)]
    test_df = df[df['item_encoded'].isin(test_items)]

    return train_df, valid_df, test_df, max_time

class MakeDataset(Dataset):
    def __init__(self, items, times, release_times, pop_histories,avg_ratings, rating_numbers, categories, stores):
        self.items = torch.tensor(items.values, dtype=torch.int)        
        self.times = torch.tensor(times.values, dtype=torch.int)
        self.release_times = torch.tensor(release_times.values, dtype=torch.int)
        self.pop_histories = [torch.tensor(h, dtype=torch.int) for h in pop_histories]
        self.avg_ratings = torch.tensor(avg_ratings.values, dtype=torch.float)
        self.rating_numbers = torch.tensor(rating_numbers.values, dtype=torch.float)
        self.categories = torch.tensor(categories.values, dtype=torch.int)
        self.stores = torch.tensor(stores.values, dtype=torch.int)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        data = {
            'item': self.items[idx],
            'time': self.times[idx],
            'release_time': self.release_times[idx],
            'pop_history': self.pop_histories[idx],            
            'avg_rating': self.avg_ratings[idx],
            'rating_number': self.rating_numbers[idx],
            'category': self.categories[idx],
            'store': self.stores[idx]
        }
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['item_encoded'], train_df['unit_time'], train_df['release_time'], train_df['pop_history'],
        train_df['avg_rating'], train_df['rating_number'], train_df['cat_encoded'], train_df['store_encoded']
    )
    valid_dataset = MakeDataset(
        valid_df['item_encoded'], valid_df['unit_time'], valid_df['release_time'], valid_df['pop_history'], valid_df['avg_rating'], valid_df['rating_number'], valid_df['cat_encoded'], valid_df['store_encoded']
    )
    test_dataset = MakeDataset(
        test_df['item_encoded'], test_df['unit_time'], test_df['release_time'], test_df['pop_history'], test_df['avg_rating'], test_df['rating_number'], test_df['cat_encoded'], test_df['store_encoded']
    )
    return train_dataset, valid_dataset, test_dataset