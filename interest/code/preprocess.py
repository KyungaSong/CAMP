import random
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle
import gc  
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from multiprocessing import Pool
from itertools import chain

tqdm.pandas()

def load_file(file_path):
    with open(file_path, 'rb') as file:
        result = pickle.load(file)                
    return result

def get_history(group):
    group_array = np.array(group)
    histories = []
    for i in range(len(group_array)):
        history = group_array[max(0, i - 128 + 1):i + 1]  
        histories.append(np.pad(history, (128 - len(history), 0), mode='constant'))    
    return histories

def calculate_ranges(group, k_m, k_s):
    k_m_delta = relativedelta(months=k_m)
    k_s_delta = relativedelta(months=k_s)
    group.set_index('timestamp', inplace=True)
    
    def get_mid_len(x):
        start_time = max(group.index.min(), x - k_m_delta)
        return group.loc[start_time:x].shape[0] - 1

    def get_short_len(x):
        start_time = max(group.index.min(), x - k_s_delta)
        return group.loc[start_time:x].shape[0] - 1

    group['mid_len'] = group.index.to_series().apply(get_mid_len)
    group['short_len'] = group.index.to_series().apply(get_short_len)
    group.reset_index(inplace=True)

    return group[['mid_len', 'short_len']]

def generate_negative_samples_for_row(all_item_ids, item_encoded, item_his_encoded_set, num_samples, item_to_cat, pop_dict, unit_time):
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

def generate_negative_samples_chunk(df_chunk, pop_dict, all_item_ids, num_samples, item_to_cat):
    chunk_neg_samples = []
    for idx in tqdm(range(len(df_chunk)), desc="Generating negative samples"):
        neg_samples = generate_negative_samples_for_row(
            all_item_ids, df_chunk.iloc[idx]['item_encoded'], df_chunk.iloc[idx]['item_his_encoded_set'], num_samples, item_to_cat, pop_dict, df_chunk.iloc[idx]['unit_time']
        )
        chunk_neg_samples.extend(neg_samples)
        if idx % 100 == 0:
            gc.collect() 
    gc.collect()  
    return chunk_neg_samples

def generate_negative_samples_vectorized_parallel(df, pop_dict, all_item_ids, num_samples, item_to_cat, num_workers=8):
    df_split = np.array_split(df, num_workers)

    print("Starting parallel processing with {} workers".format(num_workers))
    
    with Pool(num_workers) as pool:
        results = []
        for chunk in df_split:
            result = pool.apply_async(generate_negative_samples_chunk, (chunk, pop_dict, all_item_ids, num_samples, item_to_cat))
            results.append(result)
        
        neg_samples = []
        for result in results:
            try:
                chunk_neg_samples = result.get(timeout=300)  # 5 minutes timeout for each chunk
                neg_samples.extend(chunk_neg_samples)
            except TimeoutError:
                print("A chunk took too long to process and was terminated.")
                continue
    
    print("Parallel processing completed")

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

    print("Negative samples generated successfully")

    return neg_samples_df

def create_pop_dict(df_pop):
    pop_dict = {}
    for index, row in df_pop.iterrows():
        key = (row['item_encoded'], row['unit_time'])
        pop_dict[key] = {'conformity': row['conformity'], 'quality': row['quality']}
    return pop_dict

def preprocess_df(df, df_pop, config):
    df = df.copy()
    df = df.sort_values(by=['user_encoded', 'timestamp'])
    item_to_cat = df.set_index('item_encoded')['cat_encoded'].to_dict()
    pop_dict = create_pop_dict(df_pop)
    
    num_users = df['user_encoded'].max() + 1
    max_time = df["unit_time"].max()
    print("max_time", max_time)
    df = df.merge(df_pop, on=['item_encoded', 'unit_time'], how='left')

    df['item_his_encoded'] = df.groupby('user_encoded')['item_encoded'].transform(get_history)
    df['cat_his_encoded'] = df.groupby('user_encoded')['cat_encoded'].transform(get_history)
    df['con_his'] = df.groupby('user_encoded')['conformity'].transform(get_history)
    df['qlt_his'] = df.groupby('user_encoded')['quality'].transform(get_history)

    df['item_his_encoded_set'] = df['item_his_encoded'].apply(set)
    df['label'] = 1

    max_item_id = df['item_encoded'].max()
    all_item_ids = set(range(1, max_item_id + 1))

    ranges_df = df.groupby('user_encoded', group_keys=False).apply(lambda x: calculate_ranges(x, config.k_m, config.k_s), include_groups=False)
    df.reset_index(drop=True, inplace=True)
    ranges_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, ranges_df], axis=1)

    df = df[['user_encoded', 'item_encoded', 'cat_encoded', 'conformity', 'quality', 'item_his_encoded', 'item_his_encoded_set', 'cat_his_encoded', 'con_his', 'qlt_his', 'timestamp', 'unit_time', 'mid_len', 'short_len', 'label']]

    if config.data_type == 'reg':
        temp_df, test_df = train_test_split(df, test_size=0.14, random_state=42)
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif config.data_type == 'skew':
        test_size = int(len(df) * 0.2)
        max_sample_size = test_size // df['item_encoded'].nunique()
        test_df = df.groupby('item_encoded').apply(lambda x: x.sample(n=min(len(x), max_sample_size), random_state=42)).reset_index(drop=True)
        temp_df = df.drop(test_df.index)
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif config.data_type == "seq":
        train_df = df[df['unit_time'] < max_time - 1]
        rest_data = df[df['unit_time'] >= max_time - 1].reset_index(drop=True)
        val_user_set = np.random.choice(np.arange(num_users), int(num_users / 2), replace=False)
        valid_df = rest_data[rest_data['user_encoded'].isin(val_user_set)].reset_index(drop=True)
        test_df = rest_data[~rest_data['user_encoded'].isin(val_user_set)].reset_index(drop=True)
    else:
        raise ValueError("Invalid data_type. Please enter 'reg', 'skew', or 'seq'.")

    total_length = len(df)
    train_ratio = len(train_df) / total_length
    valid_ratio = len(valid_df) / total_length
    test_ratio = len(test_df) / total_length

    print(f"Train ratio: {train_ratio:.2f}, Valid ratio: {valid_ratio:.2f}, Test ratio: {test_ratio:.2f}")

    del df
    gc.collect()

    print("Generating negative samples for train dataset")
    train_neg_df = generate_negative_samples_vectorized_parallel(train_df, pop_dict, all_item_ids, config.train_num_samples, item_to_cat)
    print("Generating negative samples for valid dataset")
    valid_neg_df = generate_negative_samples_vectorized_parallel(valid_df, pop_dict, all_item_ids, config.valid_num_samples, item_to_cat)
    print("Generating negative samples for test dataset")
    # test_neg_df = generate_negative_samples_vectorized_parallel(test_df, pop_dict, all_item_ids, max_item_id, item_to_cat)
    test_neg_df = generate_negative_samples_vectorized_parallel(test_df, pop_dict, all_item_ids, config.test_num_samples, item_to_cat)

    train_df = pd.concat([train_df, train_neg_df], ignore_index=True)
    valid_df = pd.concat([valid_df, valid_neg_df], ignore_index=True)
    test_df = pd.concat([test_df, test_neg_df], ignore_index=True)
    # test_df = pd.concat([test_df, test_can_df], ignore_index=True)

    del train_neg_df, valid_neg_df, test_neg_df
    gc.collect()
    torch.cuda.empty_cache()

    return train_df, valid_df, test_df

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
        mid_len = torch.tensor(self.df['mid_len'].iloc[idx], dtype=torch.int)
        short_len = torch.tensor(self.df['short_len'].iloc[idx], dtype=torch.int)
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
            'mid_len': mid_len,
            'short_len': short_len,
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