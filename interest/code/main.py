import random
import gzip
import time
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder

from preprocess import load_dataset, preprocess_df, create_datasets
from Model import CAMP
from training_utils import train, evaluate, test

########################################################### config
random.seed(42) 
max_length = 128
time_unit = 60*60*24*1000# a day
k_m = 3*12*30*time_unit # three year
k_s = 6*30*time_unit # 6 month
batch_size = 128

# Instantiate the model
embedding_dim = 128
hidden_dim = 256
output_dim = 1

num_epochs = 10  

###################################################################### main
logging.basicConfig(filename='../../log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

dataset_name = 'sampled_Home_and_Kitchen'
processed_path = f'../../dataset/preprocessed/{dataset_name}/'

# if os.path.exists(f'{processed_path}/train_df.pkl') and os.path.exists(f'{processed_path}/valid_df.pkl') and os.path.exists(f'{processed_path}/test_df.pkl'):
#     print("Processed files already exist. Skipping dataset preparation.")
# else:
#     print("Preraing dataset......")
#     start_time = time.time()
#     df = load_dataset(f'../../dataset/{dataset_name}/{dataset_name}.csv')
#     len_df = len(df)
#     user_encoder = LabelEncoder().fit(df['user_id'])
#     item_encoder = LabelEncoder().fit(df['item_id'])
#     num_users = df['user_id'].nunique()
#     num_items = df['item_id'].nunique()

#     train_df, valid_df, test_df = preprocess_df(df, user_encoder, item_encoder, k_m, k_s, max_length)

#     if not os.path.exists(processed_path):
#         os.makedirs(processed_path)
#     date_str = datetime.now().strftime('%Y%m%d')
#     train_df.to_pickle(f'{processed_path}/train_df_{date_str}_{num_users}_{len_df}.pkl')
#     valid_df.to_pickle(f'{processed_path}/valid_df_{date_str}_{num_users}_{len_df}.pkl')
#     test_df.to_pickle(f'{processed_path}/test_df_{date_str}_{num_users}_{len_df}.pkl')
#     end_time = time.time()
#     pre_t = end_time - start_time
#     print(f"Dataset prepared in {pre_t:.2f} seconds")

print("Preraing dataset......")
start_time = time.time()
df = load_dataset(f'../../dataset/{dataset_name}/{dataset_name}.csv')
len_df = len(df)
user_encoder = LabelEncoder().fit(df['user_id'])
item_encoder = LabelEncoder().fit(df['item_id'])
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

train_df, valid_df, test_df = preprocess_df(df, user_encoder, item_encoder, k_m, k_s, max_length)

if not os.path.exists(processed_path):
    os.makedirs(processed_path)
date_str = datetime.now().strftime('%Y%m%d')
train_df.to_pickle(f'{processed_path}/train_df_{date_str}_{num_users}_{len_df}.pkl')
valid_df.to_pickle(f'{processed_path}/valid_df_{date_str}_{num_users}_{len_df}.pkl')
test_df.to_pickle(f'{processed_path}/test_df_{date_str}_{num_users}_{len_df}.pkl')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CAMP(num_users, num_items, embedding_dim, hidden_dim, output_dim).to(device)
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    
# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.001)

print("Training......")
start_time = time.time()
# Train and evaluate
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    valid_loss = evaluate(model, valid_loader, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')
end_time = time.time()
train_t = end_time - start_time
print(f"Training and Evaluation End in {train_t:.2f} seconds")

logging.info(f'Number of users: {num_users}, Number of interactions: {len_df}, Dataset preparation time: {pre_t} seconds, DataLoader loading time: {load_t} seconds, Training time: {train_t} seconds')

# Evaluate on test set
average_loss, all_top_k_items, avg_precision, avg_recall, avg_ndcg, avg_hit_rate = test(model, test_loader, device)
