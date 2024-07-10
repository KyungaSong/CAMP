import numpy as np
import pandas as pd
import pickle
import gc  
import torch
from torch.utils.data import Dataset

def load_dataset(file_path, meta = False):
    with open(file_path, 'rb') as file:
        df = pickle.load(file)         
    return df

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

def preprocess_df(df, config): 
    df = df.copy()

    df = df.sort_values(by=['item_encoded', 'unit_time'])

    min_unit_time = df.groupby('item_encoded')['unit_time'].transform('min')
    df.loc[:, 'release_time'] = min_unit_time

    max_time = df["unit_time"].max()
    print("max_time", max_time)
    time_range = list(range(0, max_time+1))

    count_per_group = df.groupby(['item_encoded', 'unit_time']).size().unstack(fill_value=0)
    count_per_group = count_per_group.reindex(columns=time_range, fill_value=0)

    df_pop = count_per_group.apply(lambda row: row.tolist(), axis=1)
    df_pop = df_pop.reset_index(name='pop_history')

    first_df = df.drop_duplicates(subset='item_encoded', keep='first')
    first_df = first_df[['item_encoded', 'release_time', 'average_rating', 'cat_encoded', 'store_encoded']]
    
    first_df = first_df.merge(df_pop, on='item_encoded', how='right')
    result_df = pd.concat([expand_time(row, max_time) for _, row in first_df.iterrows()]).reset_index(drop=True)

    train_df = result_df[result_df['unit_time'] <= max_time - 2].reset_index(drop=True)
    valid_df = result_df[result_df['unit_time'] == max_time - 1].reset_index(drop=True)
    test_df = result_df[result_df['unit_time'] == max_time].reset_index(drop=True)

    total_length = len(result_df)
    train_ratio = len(train_df) / total_length
    valid_ratio = len(valid_df) / total_length
    test_ratio = len(test_df) / total_length

    print(f"Train ratio: {train_ratio:.2f}, Valid ratio: {valid_ratio:.2f}, Test ratio: {test_ratio:.2f}")

    del df, first_df, result_df
    gc.collect()

    return train_df, valid_df, test_df, max_time

class MakeDataset(Dataset):
    def __init__(self, items, times, release_times, pop_histories, average_ratings, categories, stores):
        self.items = torch.tensor(items.values, dtype=torch.int)        
        self.times = torch.tensor(times.values, dtype=torch.int)
        self.release_times = torch.tensor(release_times.values, dtype=torch.int)
        self.pop_histories = [torch.tensor(h, dtype=torch.int) for h in pop_histories]
        self.average_ratings = torch.tensor(average_ratings.values, dtype=torch.float)
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
            'average_rating': self.average_ratings[idx],
            'category': self.categories[idx],
            'store': self.stores[idx]
        }
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['item_encoded'], train_df['unit_time'], train_df['release_time'], train_df['pop_history'],
        train_df['average_rating'], train_df['cat_encoded'], train_df['store_encoded']
    )
    valid_dataset = MakeDataset(
        valid_df['item_encoded'], valid_df['unit_time'], valid_df['release_time'], valid_df['pop_history'], valid_df['average_rating'], valid_df['cat_encoded'], valid_df['store_encoded']
    )
    test_dataset = MakeDataset(
        test_df['item_encoded'], test_df['unit_time'], test_df['release_time'], test_df['pop_history'], test_df['average_rating'], test_df['cat_encoded'], test_df['store_encoded']
    )
    return train_dataset, valid_dataset, test_dataset