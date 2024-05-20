import torch
import pandas as pd
import pickle
import argparse
import os
from torch.utils.data import DataLoader

from Model import PopPredict
from config import Config
from preprocess import load_dataset, preprocess_df, create_datasets

# 학습된 모델의 경로
model_path = '../../model/pop/sampled_Home_and_Kitchen_checkpoint_epoch_19.pt'

# 데이터셋 이름 및 경로 설정
dataset_name = 'sampled_Home_and_Kitchen'
processed_path = f'../../dataset/preprocessed/pop/{dataset_name}/'

# Config 객체 초기화
args = argparse.Namespace(
    alpha=0.7,
    batch_size=64,
    lr=0.001,
    num_epochs=20,
    time_unit=1000 * 60 * 60 * 24,
    pop_time_unit=30,
    dataset=dataset_name,
    data_preprocessed=True,
    test_only=True,
    embedding_dim=128
)
config = Config(args=args)

# NCCL 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'

# GPU 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
def load_data(dataset_name):
    with open(f'{processed_path}/train_df_pop.pkl', 'rb') as file:
        train_df = pickle.load(file)
    with open(f'{processed_path}/valid_df_pop.pkl', 'rb') as file:
        valid_df = pickle.load(file)
    with open(f'{processed_path}/test_df_pop.pkl', 'rb') as file:
        test_df = pickle.load(file)

    combined_df = pd.concat([train_df, valid_df, test_df])
    num_items = combined_df['item_encoded'].nunique()
    num_cats = combined_df['cat_encoded'].nunique()
    num_stores = combined_df['store_encoded'].nunique()
    max_time = combined_df['unit_time'].max()
    
    return train_df, valid_df, combined_df, num_items, num_cats, num_stores, max_time

# 데이터 로드
train_df, valid_df, combined_df, num_items, num_cats, num_stores, max_time = load_data(dataset_name)

# 모델 초기화 및 로드
model = PopPredict(False, config, num_items, num_cats, num_stores, max_time).to(device)

state_dict = torch.load(model_path)
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

# 현재 모델의 cat_embedding 크기를 저장된 모델의 크기에 맞게 조정
if 'cat_embedding.weight' in new_state_dict:
    old_cat_embedding_weight = new_state_dict['cat_embedding.weight']
    num_embeddings, embedding_dim = old_cat_embedding_weight.shape

    if num_embeddings != model.cat_embedding.num_embeddings:
        model.cat_embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0).to(device)
        print(f"Adjusted cat_embedding to have {num_embeddings} embeddings.")

# 나머지 매개변수 업데이트
model_state_dict = model.state_dict()
model_state_dict.update(new_state_dict)
model.load_state_dict(model_state_dict)

model.eval()

# 테스트 데이터셋 로드
_, _, test_dataset = create_datasets(train_df, valid_df, combined_df)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# 결과 저장할 리스트 초기화
pop_history_outputs = []
time_outputs = []
sideinfo_outputs = []

# 테스트 데이터셋에 대해 예측 수행
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        pop_history_output, time_output, sideinfo_output, _ = model(batch)
        
        # 결과를 리스트에 추가
        pop_history_outputs.append(pop_history_output.cpu())
        time_outputs.append(time_output.cpu())
        sideinfo_outputs.append(sideinfo_output.cpu())

pop_history_outputs = torch.cat(pop_history_outputs, dim=0)
time_outputs = torch.cat(time_outputs, dim=0)
sideinfo_outputs = torch.cat(sideinfo_outputs, dim=0)

combined_df['pop_history_output'] = pop_history_outputs.numpy()
combined_df['time_output'] = time_outputs.numpy()
combined_df['sideinfo_output'] = sideinfo_outputs.numpy()

result_path = '../../results/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

combined_df.to_pickle(f'{result_path}/test_results.pkl')
print("Results saved to", f'{result_path}/test_results.pkl')
print("combined_df\n", combined_df)
