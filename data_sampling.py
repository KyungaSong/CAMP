import pandas as pd
import numpy as np

train_path = './dataset/Home_and_Kitchen/Home_and_Kitchen.csv.gz'
# valid_path = './dataset/Home_and_Kitchen.valid.csv.gz'
# test_path = './dataset/Home_and_Kitchen.test.csv.gz'

# user_id,parent_asin,rating,timestamp,history
df = pd.read_csv(train_path, compression='gzip')
# df_valid = pd.read_csv(valid_path, compression='gzip')
# df_test = pd.read_csv(test_path, compression='gzip')

user_counts = df['user_id'].value_counts()
filtered_user_ids = user_counts[user_counts <= 129].index
df_filtered = df[df['user_id'].isin(filtered_user_ids)]
unique_user_ids = df_filtered['user_id'].unique()
print("Unique user_id count:", len(unique_user_ids))

# sampled_user_ids = np.random.choice(unique_user_ids, size=int(len(unique_user_ids) * 0.05), replace=False)
sampled_user_ids = np.random.choice(unique_user_ids, size= 512 * 256, replace=False)

df_final = df_filtered[df_filtered['user_id'].isin(sampled_user_ids)]

print(f"raw: {len(df)}, final: {len(df_final)}")

dataset = 'sampled_Home_and_Kitchen'

df_final.to_csv(f'./dataset/{dataset}/{dataset}.csv', index=False)
