import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gzip
import json

def load_dataset(file_path, side = False):    
    if side:
        # with gzip.open(file_path, 'rb') as g:
        #     df= pd.DataFrame([eval(line.decode('utf-8')) for line in g])
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz:
            data = [json.loads(line) for line in gz]
        df = pd.DataFrame(data)
        necessary_columns = {'main_category': 'genre', 'average_rating': 'avg_rating', 'store': 'director', 'parent_asin': 'item_id'}
        df = df[list(necessary_columns.keys())].rename(columns=necessary_columns)
    else:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'parent_asin': 'item_id'})          
    return df

dataset_name = 'sampled_Home_and_Kitchen'
df = load_dataset(f'../../dataset/{dataset_name}/{dataset_name}.csv')
item_encoder = LabelEncoder().fit(df['item_id'])
num_items = df['item_id'].nunique()
dataset_name = 'Home_and_Kitchen'
df_side = load_dataset(f'../../dataset/{dataset_name}/meta_{dataset_name}.jsonl.gz', side = True)
filtered_df_side = pd.merge(df[['item_id']], df_side, on='item_id')

print("filtered_df_side\n", filtered_df_side['director'].unique())

time_unit = 1000 * 60 * 60 * 24 # a day
pop_time_unit = 30 * time_unit # a month

def preprocess_df(df, pop_time_unit, df_side): 
    
    df['item_encoded'] = item_encoder.transform(df['item_id']) + 1

    min_time_all = df['timestamp'].min()
    df['unit_time'] = (df['timestamp'] - min_time_all) // pop_time_unit

    time_range = list(range(df["unit_time"].min(), df["unit_time"].max()+1))  
    count_per_group = df.groupby(['item_encoded', 'unit_time']).size().unstack(fill_value=0)
    count_per_group = count_per_group.reindex(columns=time_range, fill_value=0)
    df_result = count_per_group.apply(lambda row: row.tolist(), axis=1).reset_index()
    df_result.rename(columns={0: 'pop_history'}, inplace=True)

    min_unit_time_per_item = df.groupby('item_encoded')['unit_time'].min()
    df_pop = df_result.merge(min_unit_time_per_item.rename('release_time'), on='item_encoded')

    df_side['item_encoded'] = item_encoder.transform(df_side['item_id']) + 1


    return df_pop

df_fianl = preprocess_df(df, pop_time_unit, df_side)


# side_info 구하기 