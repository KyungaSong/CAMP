import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def load_dataset(file_path, meta = False):
    with open(file_path, 'rb') as file:
        df = pickle.load(file) 
    if meta:
        necessary_columns = ['main_category', 'average_rating', 'rating_number', 'store', 'parent_asin']
        df = df[necessary_columns].rename(columns={'main_category': 'category', 'average_rating': 'avg_rating', 'parent_asin': 'item_id'})
    else:        
        df = df.rename(columns={'parent_asin': 'item_id'})          
    return df

def encode_column(column, pad = False):
    frequencies = column.value_counts(ascending=False)
    if pad:
        mapping = pd.Series(index=frequencies.index, data=range(1, len(frequencies) + 1))
    else:
        mapping = pd.Series(index=frequencies.index, data=range(len(frequencies)))
    return column.map(mapping)

def preprocess_df(df, df_meta, config): 
    
    df['item_encoded'] = encode_column(df['item_id'], pad = True)
    df_meta['cat_encoded'] = encode_column(df['category'], pad = True)
    df_meta['store_encoded'] = encode_column(df['store'], pad = True)

    min_time_all = df['timestamp'].min()
    df['unit_time'] = (df['timestamp'] - min_time_all) // (config.pop_time_unit * config.time_unit)

    time_range = list(range(df["unit_time"].min(), df["unit_time"].max()+1))  
    count_per_group = df.groupby(['item_encoded', 'unit_time']).size().unstack(fill_value=0)
    count_per_group = count_per_group.reindex(columns=time_range, fill_value=0)
    df_pop = count_per_group.apply(lambda row: row.tolist(), axis=1)
    df_pop = df_pop.reset_index(name='pop_history')

    min_unit_time_per_item = df.groupby('item_encoded')['unit_time'].min().rename('release_time')
    df_pop = df_pop.merge(min_unit_time_per_item, on='item_encoded')

    print("df_pop:\n", df_pop)

    columns = ['item_encoded', 'pop_history', 'release_time', 'avg_rating', 'rating_number', 'cat_encoded', 'store_encoded']
    df_filtered = df_pop[columns]
    print("df_filtered: \n", df_filtered)
    print("pop history ex: \n", df_filtered['pop_history'].iloc[10])
    train_df, temp_df = train_test_split(df_filtered, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    return train_df, valid_df, test_df

class MakeDataset(Dataset):
    def __init__(self, items, pop_histories, release_times, avg_ratings, rating_numbers, categories, stores):
        self.items = torch.tensor(items, dtype=torch.long)
        self.pop_histories = [torch.tensor(h, dtype=torch.long) for h in pop_histories]
        self.release_times = torch.tensor(release_times, dtype=torch.int)
        self.avg_ratings = torch.tensor(avg_ratings, dtype=torch.long)
        self.rating_numbers = torch.tensor(rating_numbers, dtype=torch.int)
        self.categories = torch.tensor(categories, dtype=torch.long)
        self.stores = torch.tensor(stores, dtype=torch.long)

    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        data = {
            'item': self.items[idx],
            'pop_history': self.pop_histories[idx],
            'release_time': self.release_times[idx],
            'avg_rating': self.avg_ratings[idx],
            'rating_number': self.rating_numbers[idx],
            'category': self.categories[idx],
            'store': self.stores[idx]
        }
        return data

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['item_encoded'], train_df['pop_history'],
        train_df['release_time'], train_df['avg_rating'], train_df['rating_time'], train_df['cat_encoded'], train_df['store_encoded']
    )
    valid_dataset = MakeDataset(
        valid_df['item_encoded'], valid_df['pop_history'],
        valid_df['release_time'], valid_df['avg_rating'], valid_df['rating_time'], valid_df['cat_encoded'], valid_df['store_encoded']
    )
    test_dataset = MakeDataset(
        test_df['item_encoded'], test_df['pop_history'],
        test_df['release_time'], test_df['avg_rating'], test_df['rating_time'], test_df['cat_encoded'], test_df['store_encoded']
    )
    return train_dataset, valid_dataset, test_dataset