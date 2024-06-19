import random
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import pickle
import gc  
import torch
from torch.utils.data import Dataset
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
        history = group_array[max(0, i - 50 + 1):i + 1]  
        histories.append(np.pad(history, (50 - len(history), 0), mode='constant'))    
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
    return chunk_neg_samples

def generate_negative_samples_vectorized_parallel(df, pop_dict, all_item_ids, num_samples, item_to_cat, num_workers=16):
    df_split = np.array_split(df, num_workers)
    
    with Pool(num_workers) as pool:
        results = pool.starmap(generate_negative_samples_chunk, [(chunk, pop_dict, all_item_ids, num_samples, item_to_cat) for chunk in df_split])
    
    neg_samples = list(chain.from_iterable(results))

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

def split_list(data, n):
    """Splits data into n chunks."""
    k, m = divmod(len(data), n)
    return [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def generate_test_samples_for_user_group(user_group, all_item_ids, item_to_cat, pop_dict):
    samples = []
    for user_encoded, group in user_group:
        try:
            group = group.sort_values(by='timestamp')

            item_his_encoded_set = set(chain.from_iterable(group['item_his_encoded']))
            user_items = set(group['item_encoded'])
            candidate_items = list(all_item_ids - item_his_encoded_set - user_items)
            
            for item in candidate_items:
                key = (item, group['unit_time'].iloc[-1])
                if key in pop_dict:
                    pop_data = pop_dict[key]
                    conformity = pop_data['conformity']
                    quality = pop_data['quality']
                    cat_encoded = item_to_cat.get(item, 0)
                    samples.append({
                        'user_encoded': user_encoded,
                        'item_encoded': item,
                        'cat_encoded': int(cat_encoded),
                        'conformity': conformity,
                        'quality': quality,
                        'item_his_encoded_set': item_his_encoded_set,
                        'unit_time': group['unit_time'].iloc[-1],
                        'mid_len': group['mid_len'].iloc[-1],
                        'short_len': group['short_len'].iloc[-1],
                        'label': 0,
                        'item_his_encoded': group['item_his_encoded'].iloc[-1],
                        'cat_his_encoded': group['cat_his_encoded'].iloc[-1],
                        'con_his': group['con_his'].iloc[-1],
                        'qlt_his': group['qlt_his'].iloc[-1]
                    })
        except Exception as e:
            print(f"Error processing user {user_encoded}: {e}")
    return samples


def generate_test_samples_chunk(user_groups_chunk, all_item_ids, item_to_cat, pop_dict):
    chunk_samples = []
    for user_group in tqdm(user_groups_chunk, desc="Generating test samples"):
        try:
            samples = generate_test_samples_for_user_group([user_group], all_item_ids, item_to_cat, pop_dict)
            chunk_samples.extend(samples)
        except Exception as e:
            print(f"Error processing user_group: {user_group[0]}. Exception: {e}")
    gc.collect()
    return chunk_samples

def generate_test_samples_vectorized_parallel(df, all_item_ids, item_to_cat, pop_dict, num_workers=16):
    user_groups = list(df.groupby('user_encoded'))
    user_groups_split = split_list(user_groups, num_workers)

    results = []
    with Pool(num_workers) as pool:
        for chunk in user_groups_split:
            result = pool.apply_async(wrapper_generate_test_samples_chunk, args=(chunk, all_item_ids, item_to_cat, pop_dict))
            results.append(result)
        
        test_samples = []
        for result in results:
            test_samples.extend(result.get())  
            gc.collect() 

    test_samples_df = pd.DataFrame(test_samples)

    return test_samples_df

def wrapper_generate_test_samples_chunk(chunk, all_item_ids, item_to_cat, pop_dict):
    try:
        return generate_test_samples_chunk(chunk, all_item_ids, item_to_cat, pop_dict)
    except Exception as e:
        print(f"Error in chunk: {chunk}. Exception: {e}")
        return []
    
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

    # if config.dataset == 'MovieLens_1M': # fix
    #     train_df = df[df['unit_time'] < 8].reset_index(drop=True)
    #     valid_df = df[df['unit_time'] == 8].reset_index(drop=True)
    #     test_df = df[df['unit_time'] >= 9].reset_index(drop=True)
    # elif config.dataset == 'Sports_and_Outdoors': # fix
    #     train_df = df[df['unit_time'] < max_time - 3].reset_index(drop=True)
    #     valid_df = df[(df['unit_time'] < max_time - 2) & (df['unit_time'] >= max_time - 3)].reset_index(drop=True)
    #     test_df = df[df['unit_time'] >= max_time -2].reset_index(drop=True)  
    # elif config.dataset == '14_Toys': # fix
    #     train_df = df[df['unit_time'] <= max_time - 2].reset_index(drop=True)
    #     rest_df = df[df['unit_time'] >= max_time - 1].reset_index(drop=True)
    #     valid_df = pd.DataFrame()
    #     test_df = pd.DataFrame()
    #     grouped = rest_df.groupby('item_encoded')
    #     for _, group in grouped:
    #         if len(group) == 1:
    #             valid_df = pd.concat([valid_df, group])
    #         else:
    #             valid, test = train_test_split(group, test_size=0.5, random_state=42)
    #             valid_df = pd.concat([valid_df, valid])
    #             test_df = pd.concat([test_df, test])
    #     valid_df = valid_df.reset_index(drop=True)
    #     test_df = test_df.reset_index(drop=True)
    # else: # sampled_Toys fix
    #     train_df = df[df['unit_time'] <= max_time - 2].reset_index(drop=True)
    #     valid_df = df[df['unit_time'] == max_time - 1].reset_index(drop=True)
    #     test_df = df[df['unit_time'] == max_time].reset_index(drop=True)
    if config.data_type == 'reg':
        temp_df, test_df = train_test_split(df, test_size=0.14, random_state=42)
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif config.data_type == 'skew':
        test_size = int(len(df) * 0.2)
        max_sample_size = test_size // df['item_encoded'].nunique()
        test_df = df.groupby('item_encoded').apply(lambda x: x.sample(n=min(len(x), max_sample_size), random_state=42)).reset_index(drop=True)
        # if len(test_df) < test_size:
        #     remaining_samples = test_size - len(test_df)
        #     additional_samples = df.drop(test_df.index).sample(n=remaining_samples, random_state=42)
        #     test_df = pd.concat([test_df, additional_samples]).reset_index(drop=True)
        temp_df = df.drop(test_df.index)
        train_df, valid_df = train_test_split(temp_df, test_size=0.1, random_state=42)
    elif config.data_type == "seq":
        df = df.sort_values(by=['user_encoded', 'timestamp'])        
        train_list = []
        valid_list = []
        test_list = []
        for user, user_df in df.groupby('user_encoded'):
            if len(user_df) >= 2:
                test_list.append(user_df.iloc[-1:])
                valid_list.append(user_df.iloc[-2:-1])
                train_list.append(user_df.iloc[:-2])
            elif len(user_df) == 1:
                test_list.append(user_df)
            
        train_df = pd.concat(train_list)
        valid_df = pd.concat(valid_list)
        test_df = pd.concat(test_list)
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
    # print("Generating negative samples for test dataset")
    # test_neg_df = generate_negative_samples_vectorized_parallel(test_df, pop_dict, all_item_ids, config.test_num_samples, item_to_cat)
    print("Generating test samples for test dataset")
    test_can_df = generate_test_samples_vectorized_parallel(test_df, all_item_ids, item_to_cat, pop_dict)

    train_df = pd.concat([train_df, train_neg_df], ignore_index=True)
    valid_df = pd.concat([valid_df, valid_neg_df], ignore_index=True)
    # test_df = pd.concat([test_df, test_neg_df], ignore_index=True)
    test_df = pd.concat([test_df, test_can_df], ignore_index=True)

    del train_neg_df, valid_neg_df, test_neg_df
    gc.collect()
    torch.cuda.empty_cache()

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
    print("making train dataset")
    train_dataset = MakeDataset(
        train_df['user_encoded'], train_df['item_encoded'], train_df['cat_encoded'], train_df['conformity'], train_df['quality'],
        train_df['item_his_encoded'], train_df['cat_his_encoded'],  train_df['con_his'], train_df['qlt_his'], 
        train_df['mid_len'], train_df['short_len'], train_df['label']
    )
    print("making valid dataset")
    valid_dataset = MakeDataset(
        valid_df['user_encoded'], valid_df['item_encoded'], valid_df['cat_encoded'], valid_df['conformity'], valid_df['quality'],
        valid_df['item_his_encoded'], valid_df['cat_his_encoded'], valid_df['con_his'], valid_df['qlt_his'], 
        valid_df['mid_len'], valid_df['short_len'], valid_df['label']
    )
    print("making test dataset")
    test_dataset = MakeDataset(
        test_df['user_encoded'], test_df['item_encoded'], test_df['cat_encoded'], test_df['conformity'], test_df['quality'],
        test_df['item_his_encoded'], test_df['cat_his_encoded'], test_df['con_his'], test_df['qlt_his'], 
        test_df['mid_len'], test_df['short_len'], test_df['label']
    )   
    print("create datasets done!") 
    return train_dataset, valid_dataset, test_dataset

# class CustomDataset(Dataset):
#     def __init__(self, df):
#         self.df = df

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         return {
#             'user': torch.tensor(row['user_encoded'], dtype=torch.long),
#             'item': torch.tensor(row['item_encoded'], dtype=torch.long),
#             'cat': torch.tensor(row['cat_encoded'], dtype=torch.long),
#             'con': torch.tensor(row['conformity'], dtype=torch.float),
#             'qlt': torch.tensor(row['quality'], dtype=torch.float),
#             'item_his': torch.tensor(row['item_his_encoded'], dtype=torch.long),
#             'cat_his': torch.tensor(row['cat_his_encoded'], dtype=torch.long),
#             'con_his': torch.tensor(row['con_his'], dtype=torch.float),
#             'qlt_his': torch.tensor(row['qlt_his'], dtype=torch.float),
#             'mid_len': torch.tensor(row['mid_len'], dtype=torch.int),
#             'short_len': torch.tensor(row['short_len'], dtype=torch.int),
#             'label': torch.tensor(row['label'], dtype=torch.long)
#         }

# class CustomIterableDataset(IterableDataset):
#     def __init__(self, df):
#         self.df = df

#     def preprocess_row(self, row):
#         return {
#             'user': torch.tensor(row['user_encoded'], dtype=torch.long),
#             'item': torch.tensor(row['item_encoded'], dtype=torch.long),
#             'cat': torch.tensor(row['cat_encoded'], dtype=torch.long),
#             'con': torch.tensor(row['conformity'], dtype=torch.float),
#             'qlt': torch.tensor(row['quality'], dtype=torch.float),
#             'item_his': torch.tensor(row['item_his_encoded'], dtype=torch.long),
#             'cat_his': torch.tensor(row['cat_his_encoded'], dtype=torch.long),
#             'con_his': torch.tensor(row['con_his'], dtype=torch.float),
#             'qlt_his': torch.tensor(row['qlt_his'], dtype=torch.float),
#             'mid_len': torch.tensor(row['mid_len'], dtype=torch.int),
#             'short_len': torch.tensor(row['short_len'], dtype=torch.int),
#             'label': torch.tensor(row['label'], dtype=torch.long)
#         }

#     def __iter__(self):
#         for _, row in self.df.iterrows():
#             yield self.preprocess_row(row)

#     def __len__(self):
#         return len(self.df)

# def create_datasets(train_df, valid_df, test_df):
#     train_dataset = CustomDataset(train_df)
#     valid_dataset = CustomIterableDataset(valid_df)
#     test_dataset = CustomIterableDataset(test_df)

#     return train_dataset, valid_dataset, test_dataset
