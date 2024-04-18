import math
import os
import gzip
import json
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils.data as data
from collections import Counter

import Config

random.seed(2023)

generate_traditional_set = False  # If generate dataset for traditional model

from matplotlib import pyplot as plt

def sta_top_pop(config, dict_item_pop, dict_item_idx, test_start_time):
    cut_off = [1, 3, 6, 12, 5, 10, 20]
    list_item_in_cutoff_range = np.zeros((len(dict_item_idx), len(cut_off) + 1))

    for key in tqdm(dict_item_pop.keys()):
        item_pop = dict_item_pop[key]
        if key in dict_item_idx:
            idx = dict_item_idx[key]

            item_pop_arr = np.array(item_pop)
            for i, co in enumerate(cut_off):
                list_item_in_cutoff_range[idx][i] = np.sum(item_pop_arr[test_start_time - co:test_start_time])

            list_item_in_cutoff_range[idx][-1] = item_pop_arr[test_start_time]

    path = './result/' + config.dataset + '_{}_toppop.pop'
    
    for cut_off_num, co in enumerate(cut_off + ['gt']):
        with open(path.format(co), 'w') as f:
            for i, data in enumerate(list_item_in_cutoff_range[:, cut_off_num]):
                f.write('{} {}\n'.format(i, data))



def get_dict_a1_a2(df, json_root, dataset_name, a1, a2):
    dict_a1_a2 = {}
    unique_a1_count = df[a1].nunique()
    print(f'num_{a1}: {unique_a1_count}')

    json_filename = f"{dataset_name.split('.')[0]}_dict_{a1}_{a2}.json"
    json_path = os.path.join(json_root, json_filename)

    if os.path.exists(json_path):
        print(f'dict_{a1}_{a2} json found')
        with open(json_path, 'r') as file:
            inter_json = json.load(file)
        dict_a1_a2 = inter_json[f'dict_{a1}_{a2}']
    else:
        print(f'dict_{a1}_{a2} json not found')
        grouped_df = df.groupby(a1)[a2].apply(list).to_dict()
        dict_a1_a2 = grouped_df
        with open(json_path, 'w') as file:
            json.dump({f'dict_{a1}_{a2}': dict_a1_a2}, file)

    return dict_a1_a2


def filter_user_item_limit(df, user_limit, item_limit):
    # 사용자 및 아이템별 리뷰 수 계산
    user_counts = df['user'].value_counts()
    item_counts = df['item'].value_counts()

    # 최소 리뷰 수를 충족하는 사용자 및 아이템 필터링
    users_to_keep = user_counts[user_counts >= user_limit].index
    items_to_keep = item_counts[item_counts >= item_limit].index

    # 필터링된 사용자 및 아이템을 가진 데이터만 선택
    filtered_df = df[df['user'].isin(users_to_keep) & df['item'].isin(items_to_keep)]

    # 반복적 필터링을 위해 설정
    while len(filtered_df) < len(df):
        df = filtered_df
        user_counts = df['user'].value_counts()
        item_counts = df['item'].value_counts()
        users_to_keep = user_counts[user_counts >= user_limit].index
        items_to_keep = item_counts[item_counts >= item_limit].index
        filtered_df = df[df['user'].isin(users_to_keep) & df['item'].isin(items_to_keep)]

    return filtered_df


def list_to_csv(data_list, csv_columns):
    # DataFrame 생성 시 바로 data_list와 csv_columns를 사용
    df = pd.DataFrame(data_list, columns=csv_columns)
    return df


def make_dataset(config: Config.Config):
    main_path = config.main_path
    processed_path = config.processed_path
    train_path = config.train_path
    valid_path = config.valid_path
    test_path = config.test_path
    side_info_path = config.side_info_path
    info_path = config.info_path
    dataset = config.dataset
    time_unit = config.time_unit
    pop_time_unit = config.pop_time_unit
    pop_history_length = config.pop_history_length
    pos_item_pop_limit = config.pos_item_pop_limit
    test_time_range = config.test_time_range
    neg_item_num = config.neg_item_num
    is_douban = config.is_douban
    douban_rate_limit = config.douban_rate_limit
    do_sta = config.do_sta

    dataset_paths = [train_path, valid_path, test_path]
    if all(os.path.exists(path) for path in dataset_paths) and not do_sta:
        print("Processed dataset found.")
        return
    else:
        print('Processed dataset not found. Processing begin:')

    ############################################ read dataset ##################################################    
    if 'Amazon' in main_path:    
        data = []
        with gzip.open(main_path + dataset, 'rb') as g:
            for line in g:
                # eval을 사용하여 문자열을 파이썬 객체로 변환
                data.append(eval(line))
        df = pd.DataFrame(data)
        df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'unixReviewTime': 'time', 'overall': 'rate'})
        df.drop(columns=['reviewerName', 'helpful', 'reviewText', 'summary', 'reviewTime'], inplace=True)
        print('len_df: {}'.format(len(df)))
    else:
        # Douban -> MovieLens 데이터셋에 맞게 수정
        # ['Bad', 'Reply', 'MovieID', 'UserID', 'ReviewID', 'Rate', 'Time', 'Good']
        df = pd.read_csv(main_path + dataset, index_col=0, header=0, dtype=str)
        df = df.rename(columns={'UserID': 'user', 'MovieID': 'item', 'Time': 'time', 'Rate': 'rate'})
        df.drop(columns=['Good', 'Bad', 'Reply', 'ReviewID'], inplace=True)
        # 평점 필터링
        df['rate'] = pd.to_numeric(df['rate'])
        df = df[df['rate'] >= douban_rate_limit]
        print('Original Length: {}, Length after filter rate: {}'.format(len(data_list), len(data_filter_rate)))

    # drop duplicate
    df.drop_duplicates(subset=['user', 'item'], keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # filter user_limit and item_limit
    df = filter_user_item_limit(df, config.user_limit, config.item_limit)

    # Find first inter time of item as release time
    df['time'] = df['time'].astype(int)  
    min_time_all = df['time'].min()
    df['unit_time'] = (df['time'] - min_time_all) // (time_unit * pop_time_unit)

    # calculate the first interaction time and maximum time by item
    dict_item_time = get_dict_a1_a2(df, processed_path, dataset, 'item', 'unit_time')
    max_time = df['unit_time'].max()

    # remove items for a period less than 2 * test_time_range + 1
    dict_item_time = {k: v for k, v in dict_item_time.items() if max(v) - min(v) >= 2 * test_time_range + 1}
    # calculate the first interaction time by item
    dict_item_time_release = {k: min(v) for k, v in dict_item_time.items()}

    ########################################### Statistic Item pop ##################################################

    dict_item_pop = {}
    for item, item_time in dict_item_time.items():
        time_counter = Counter(item_time)
        dict_item_pop[item] = [time_counter.get(i, 0) for i in range(max_time + 1)]

    ###################################### Process Side information ##################################################

    if is_douban:
        df_side_info = pd.read_csv(side_info_path, dtype=str)
        necessary_columns = {'Director': 'director', 'Actor': 'actor', 'Genre': 'genre', 'MovieID': 'name'}
    else:
        with gzip.open(side_info_path, 'rb') as g:
            df_side_info = pd.DataFrame([eval(line.decode('utf-8')) for line in g])
        necessary_columns = {'categories': 'genre', 'asin': 'name'}

    # 컬럼 선택 및 이름 변경
    df_side_info = df_side_info[list(necessary_columns.keys())].rename(columns=necessary_columns)
    print("length of df_side_info:", len(df_side_info))
    print(df_side_info.head())
    print()

    def process_side_info(df_side_info, dict_item_time, config):
        # Filter and deduplicate entries based on 'name'
        df_side_info = df_side_info[df_side_info['name'].isin(dict_item_time)].drop_duplicates('name').reset_index(drop=True)
        
        # Validate the filtered data
        if len(df_side_info) != len(dict_item_time):
            print(f'len(df_side_info): {len(df_side_info)}')
            print(f'len(dict_item_time): {len(dict_item_time)}')
            print('find item without side information or other error')
            return

        # Initialize dictionaries for mapping and counters for IDs
        dict_side_info = {}
        dict_director, dict_actor, dict_genre = {'padding': 0}, {'padding': 0}, {'padding': 0}
        num_director, num_actor, num_genre, max_genre = 1, 1, 1, 0

        def add_padding_or_truncate(list_items, target_length, pad_value='padding'):
            """Add padding to or truncate the list to ensure it has the target length."""
            return list_items[:target_length] + [pad_value] * (target_length - len(list_items))

        # Process side information based on configuration
        for i, row in df_side_info.iterrows():
            genre = row['genre'][0]  # Assuming 'genre' is a list of lists, and we're interested in the first list
            if not config.is_douban:
                # Simplify the category list to the first three elements and join them with '_'
                genre = ['_'.join(genre[:3])] if genre else ['padding']

            genre_ids = [dict_genre.setdefault(g, len(dict_genre)) for g in genre]
            max_genre = max(max_genre, len(genre_ids))
            
            # Only process director and actor for douban config
            if config.is_douban:
                director_id = [dict_director.setdefault(director[0], len(dict_director))]
                actor_ids = [dict_actor.setdefault(a, len(dict_actor)) for a in actor]
                dict_side_info[row['name']] = [genre_ids, director_id, actor_ids]
            else:
                dict_side_info[row['name']] = [genre_ids]

        # Padding genres to match the length of the longest genre list
        if config.is_douban:
            for key in dict_side_info:
                dict_side_info[key][0] += [0] * (max_genre - len(dict_side_info[key][0]))
                dict_side_info[key] = str(dict_side_info[key])

        print(f'num_genre: {num_genre}, num_actor: {num_actor}, num_director: {num_director}')
        return dict_side_info, dict_director, dict_actor, dict_genre

    # Example usage
    dict_side_info, dict_director, dict_actor, dict_genre = process_side_info(df_side_info, dict_item_time, config)

    ###################################### generate train/valid/test set ##############################################
    def update_pop_history(item_pop, valid_start_time, max_time):
        """인기도 히스토리를 업데이트하고 유효한 길이를 반환하는 함수"""
        pop_history = [item_pop[t] if t < len(item_pop) else 0 for t in range(valid_start_time)]
        valid_pop_len = len(pop_history)
        pop_history.extend([-1] * (max_time - valid_pop_len + 1))
        return pop_history, valid_pop_len

    def append_dataset(dataset, item_idx, item_time_release, side_info, time_now, pop_history, pop_gt, valid_pop_len):
        """데이터셋에 새로운 데이터 포인트를 추가하는 함수"""
        dataset.append({
            'item_idx': item_idx,
            'item_time_release': item_time_release,
            'side_info': side_info,
            'time_now': time_now,
            'pop_history': pop_history,
            'pop_gt': pop_gt,
            'valid_pop_len': valid_pop_len
    })

    # 초기 설정
    train_dataset, valid_dataset, test_dataset = [], [], []
    dict_item_idx = {}
    num_item = 0

    # 시간 범위 설정
    valid_start_time = max_time - 2 * test_time_range + 1
    test_start_time = max_time - test_time_range + 1

    # 데이터셋 구성
    for item, item_pop in dict_item_pop.items():
        item_time_release = dict_item_time_release[item]
        side_info = dict_side_info.get(item, [[0]])
        
        if item_time_release >= valid_start_time:
            continue  # 유효하지 않은 경우 건너뛰기

        if item not in dict_item_idx:
            dict_item_idx[item] = num_item
            num_item += 1
        
        for time_now in range(max_time + 1):
            pop_history, valid_pop_len = update_pop_history(item_pop, item_time_release, max_time)
            pop_gt = item_pop[time_now]
            
            if time_now < valid_start_time:
                append_dataset(train_dataset, dict_item_idx[item], item_time_release, side_info, time_now, pop_history, pop_gt, valid_pop_len)
            elif time_now < test_start_time:
                append_dataset(valid_dataset, dict_item_idx[item], item_time_release, side_info, time_now, pop_history, pop_gt, valid_pop_len)
            else:
                append_dataset(test_dataset, dict_item_idx[item], item_time_release, side_info, time_now, pop_history, pop_gt, valid_pop_len)
                
    # 데이터셋을 DataFrame으로 변환하고 CSV 파일로 저장
    def save_datasets(datasets, names):
        for dataset, name in zip(datasets, names):
            df = pd.DataFrame(dataset)
            df.to_csv(name, index=False)
            print(f'{name} set saved. Length: {len(df)}, Unique items: {df["item_idx"].nunique()}, Num zero pop: {df["pop_gt"].eq(0).sum()}')

    save_datasets([train_dataset, valid_dataset, test_dataset], [train_path, valid_path, test_path])


    ######################################### save dataset info #################################################

    # 메타 정보를 JSON 형식으로 저장
    meta_info = {
        "num_item": int(num_item), 
        "max_time": int(max_time),
        "dict_item_idx": dict_item_idx,
        "dict_item_pop": dict_item_pop,
        "dict_item_time_release": dict_item_time_release,
        "category_counts": [len(dict_genre), len(dict_director), len(dict_actor)],
        "dict_side_info": dict_side_info
    }

    with open(info_path, 'w') as f:
        json.dump(meta_info, f, indent=4)

    generate_tra_set(config, df, max_time, dict_item_idx)
    return

def write_dataset_to_file(dataset, filepath):
    """데이터셋을 파일에 저장하는 함수"""
    with open(filepath, 'w') as f:
        for line in dataset:
            f.write(f"{' '.join(map(str, line))}\n")

def generate_tra_set(config: Config.Config, df, max_time, dict_item_idx):
    test_time_range = config.test_time_range
    valid_start_time = max_time - 2 * test_time_range + 1
    test_start_time = max_time - test_time_range + 1
    df_filtered = df[df['item'].isin(dict_item_idx)]

    list_train, list_valid, list_test = [], [], []
    dict_user_idx = {}
    
    for _, row in df_filtered.iterrows():
        user_idx = dict_user_idx.setdefault(row['user'], len(dict_user_idx))
        line = [user_idx, dict_item_idx[row['item']], row['rate'], row['time']]
        
        if row['unit_time'] in range(valid_start_time, test_start_time):
            list_valid.append(line)
        elif row['unit_time'] in range(test_start_time, max_time + 1):
            list_test.append(line)
        else:
            list_train.append(line)

    # 데이터셋 저장 경로 생성
    os.makedirs(config.processed_path, exist_ok=True)
    
    # 데이터셋 저장
    train_tri_path = os.path.join(config.processed_path, f"{config.dataset.split('.')[0]}_tra_train.txt")
    valid_tri_path = os.path.join(config.processed_path, f"{config.dataset.split('.')[0]}_tra_validate.txt")
    test_tri_path = os.path.join(config.processed_path, f"{config.dataset.split('.')[0]}_tra_test.txt")

    write_dataset_to_file(list_train, train_tri_path)
    write_dataset_to_file(list_valid, valid_tri_path)
    write_dataset_to_file(list_test, test_tri_path)

    print(f"Train set saved. Length: {len(list_train)}, Unique users: {len(set(line[0] for line in list_train))}, Unique items: {len(set(line[1] for line in list_train))}")
    print(f"Valid set saved. Length: {len(list_valid)}, Unique users: {len(set(line[0] for line in list_valid))}, Unique items: {len(set(line[1] for line in list_valid))}")
    print(f"Test set saved. Length: {len(list_test)}, Unique users: {len(set(line[0] for line in list_test))}, Unique items: {len(set(line[1] for line in list_test))}")

    with open(train_tri_path, 'w') as f:
        for line in list_train:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    with open(valid_tri_path, 'w') as f:
        for line in list_valid:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    with open(test_tri_path, 'w') as f:
        for line in list_test:
            f.write('{} {} {} {}\n'.format(line[0], line[1], line[2], line[3]))
    print('Train tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_train), len(set([i[0] for i in list_train])), len(set([i[1] for i in list_train]))))
    print('Valid tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_valid), len(set([i[0] for i in list_valid])), len(set([i[1] for i in list_valid]))))
    print('Test tra set done, len: {}, num_user: {}, num_item: {}'
          .format(len(list_test), len(set([i[0] for i in list_test])), len(set([i[1] for i in list_test]))))
    return


def loaded_json(config):
    if not config.json_loaded:
        with open(config.info_path, 'r') as f:
            data = json.load(f)
            # num_item, max_time, dict_item_idx, dict_item_pop, dict_item_time_release, num_side_info, dict_side_info
            config.num_item = data.get("num_item", 0)
            config.max_time = data.get("max_time", 0)
            config.dict_item_idx = data.get("dict_item_idx", {})
            config.dict_item_pop = data.get("dict_item_pop", {})
            config.dict_item_time_release = data.get("dict_item_time_release", {})
            config.num_side_info = data.get("category_counts", [])
            config.dict_side_info = data.get("dict_side_info", {})

        config.json_loaded = True


def pop_func(x):
    if x>0:
        return math.log(x + 1)
    else:
        return x


class Data(data.Dataset):
    def __init__(self, config:Config.Config, set_type):
        super(Data, self).__init__()
        self.config = config
        self.set_type = set_type
        make_dataset(config)
        loaded_json(config)
        self.data = self.load_dataset()
        self.process_pop()

        self.data_ori = self.data

    def process_pop(self):
        self.data['pop_gt'] = list(map(pop_func, self.data['pop_gt']))
        self.data['pop_history'] = [list(map(pop_func, i)) for i in self.data['pop_history']]

    def load_dataset(self):
        if self.set_type == 'Train':
            data_path = self.config.train_path
        elif self.set_type == 'Valid':
            data_path = self.config.valid_path
        elif self.set_type == 'Test':
            data_path = self.config.test_path
        else:
            print('Dataset type error!')
            exit()
        df = pd.read_csv(data_path, header=0, index_col=0,
                         dtype={'item': int, 'time_release': int, 'side_info': str,
                                'time': int, 'pop_history': str, 'pop_gt': int, 'valid_pop_len': int})

        df['side_info'] = list(map(eval, df['side_info']))
        df['pop_history'] = list(map(eval, df['pop_history']))
        return df
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # ['item', 'time_release', 'side_info', 'time', 'pop_history', 'pop_gt', 'valid_pop_len']
        return [self.data['item'][idx], self.data['time_release'][idx], self.data['side_info'][idx],
                self.data['time'][idx], self.data['pop_history'][idx], self.data['pop_gt'][idx],
                self.data['valid_pop_len'][idx]]