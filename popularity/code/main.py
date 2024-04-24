import random
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from preprocess import load_dataset, preprocess_df, create_datasets

from torch.utils.data import DataLoader

########################################################### config
random.seed(42) 
max_length = 128
time_unit = 1000 * 60 * 60 * 24 # a day
pop_time_unit = 30 * time_unit # a month
batch_size = 128

print("Preraing dataset......")
start_time = time.time()
dataset_name = 'toy_Home_and_Kitchen'
df = load_dataset(f'../../dataset/{dataset_name}/{dataset_name}.csv')

dataset_name = 'Home_and_Kitchen'
df_side = load_dataset(f'../../dataset/{dataset_name}/meta_{dataset_name}.jsonl.gz', side = True)
filtered_side = df_side[df_side['item_id'].isin(df['item_id'])].copy()

item_encoder = LabelEncoder().fit(df['item_id'])
cat_encoder = LabelEncoder().fit(filtered_side['category'])
store_encoder = LabelEncoder().fit(filtered_side['store'])

train_df, valid_df, test_df = preprocess_df(df, filtered_side, item_encoder, cat_encoder, store_encoder, pop_time_unit)
end_time = time.time()
pre_t = end_time - start_time
print(f"Dataset prepared in {pre_t:.2f} seconds")

print("Data Loading......")
train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
end_time = time.time()
load_t = end_time - start_time
print(f"Dataset prepared in {load_t:.2f} seconds")