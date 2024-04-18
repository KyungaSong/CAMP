import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
import gzip
from tqdm import tqdm
import time
import logging

################################################################## Module
class LongTermInterestModule(nn.Module):
    def __init__(self, embedding_dim):
        super(LongTermInterestModule, self).__init__()
        self.W_l = nn.Parameter(torch.randn(embedding_dim, embedding_dim))
        self.bn = nn.BatchNorm1d(embedding_dim)  
        self.dropout = nn.Dropout(0.5)  
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),  
            nn.Dropout(0.5),                
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, history_embeds, user_embed):
        h = torch.matmul(history_embeds, self.W_l)
        h = self.bn(h)  
        h = self.dropout(h)  
        user_embed_expanded = user_embed.unsqueeze(1).expand(-1, h.size(1), -1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_l = torch.sum(a.unsqueeze(2) * history_embeds, dim=1)
        return z_l

class MidTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MidTermInterestModule, self).__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.W_m = nn.Parameter(torch.randn(hidden_dim, embedding_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, 1)
        )

    def forward(self, history_embeds, user_embed):
        o, _ = self.rnn(history_embeds)
        o = o.permute(0, 2, 1)  
        o = self.bn(o)
        o = o.permute(0, 2, 1)  
        o = self.dropout(o)
        h = torch.matmul(o, self.W_m)
        user_embed_expanded = user_embed.unsqueeze(1).expand(-1, h.size(1), -1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_m = torch.sum(a.unsqueeze(2) * history_embeds, dim=1)
        return z_m


class ShortTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ShortTermInterestModule, self).__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)  
        self.dropout = nn.Dropout(0.5)  
        self.W_s = nn.Parameter(torch.randn(hidden_dim, embedding_dim))
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),  
            nn.Dropout(0.5),               
            nn.Linear(embedding_dim, 1)
        )
    
    def forward(self, history_embeds, user_embed):
        o, _ = self.rnn(history_embeds)
        o = o.permute(0, 2, 1)  
        o = self.bn(o)  
        o = o.permute(0, 2, 1) 
        o = self.dropout(o)  
        h = torch.matmul(o, self.W_s)
        user_embed_expanded = user_embed.unsqueeze(1).expand(-1, h.size(1), -1)
        combined = h * user_embed_expanded
        alpha = self.mlp(combined).squeeze(2)
        a = torch.softmax(alpha, dim=1)
        z_s = torch.sum(a.unsqueeze(2) * history_embeds, dim=1)
        return z_s

    
def long_term_interest_proxy(history_item_embeds):
    """
    Calculate the long-term interest proxy.
    """
    p_l_t = torch.mean(history_item_embeds, dim=1)
    return p_l_t

def mid_term_interest_proxy(history_item_embeds, mid_lens):
    """
    Calculate the mid-term interest proxy using masking for variable lengths.
    """
    device = history_item_embeds.device
    max_len = history_item_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(mid_lens), max_len) < mid_lens.unsqueeze(1).to(device)
    
    masked_history = history_item_embeds * mask.unsqueeze(-1).type_as(history_item_embeds)
    valid_counts = mask.sum(1, keepdim=True)

    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_m_t = masked_history.sum(1) / safe_valid_counts.type_as(history_item_embeds)
    p_m_t = torch.nan_to_num(p_m_t, nan=0.0)

    return p_m_t

def short_term_interest_proxy(history_item_embeds, short_lens):
    """
    Calculate the short-term interest proxy.
    """
    device = history_item_embeds.device
    max_len = history_item_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(short_lens), max_len) < short_lens.unsqueeze(1).to(device)
    
    masked_history = history_item_embeds * mask.unsqueeze(-1).type_as(history_item_embeds)
    valid_counts = mask.sum(1, keepdim=True)

    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_s_t = masked_history.sum(1) / safe_valid_counts.type_as(history_item_embeds)
    p_s_t = torch.nan_to_num(p_s_t, nan=0.0)

    return p_s_t


def bpr_loss(a, positive, negative):
    """
    Simplified BPR loss without the logarithm, for contrastive tasks.
    
    Parameters:
    - a: The embedding vector (z_l, z_m, or z_s).
    - positive: The positive proxy or representation (p_l or z_l for long-term, and similarly for mid and short-term).
    - negative: The negative proxy or representation (p_m or z_m for long-term, and similarly for mid and short-term).
    """
    pos_score = torch.sum(a * positive, dim=1)  
    neg_score = torch.sum(a * negative, dim=1)
    return F.softplus(neg_score - pos_score)

def calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s):
    """
    Calculate the overall contrastive loss L_con for a user at time t,
    which is the sum of L_lm (long-mid term contrastive loss) and L_ms (mid-short term contrastive loss).
    """
    # Loss for the long-term and mid-term interests pair
    L_lm = bpr_loss(z_l, p_l, p_m) + bpr_loss(p_l, z_l, z_m) + \
           bpr_loss(z_m, p_m, p_l) + bpr_loss(p_m, z_m, z_l)
    
    # Loss for the mid-term and short-term interests pair
    L_ms = bpr_loss(z_m, p_m, p_s) + bpr_loss(p_m, z_m, z_s) + \
           bpr_loss(z_s, p_s, p_m) + bpr_loss(p_s, z_s, z_m)
    
    # Overall contrastive loss
    L_con = L_lm + L_ms
    return L_con

class InterestFusionModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(InterestFusionModule, self).__init__()
        self.gru_l = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.gru_m = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
        # BatchNorm layers after GRU outputs
        self.bn_l = nn.BatchNorm1d(hidden_dim)
        self.bn_m = nn.BatchNorm1d(hidden_dim)

        # Dropout layers after BatchNorm
        self.dropout_l = nn.Dropout(0.5)
        self.dropout_m = nn.Dropout(0.5)

        self.mlp_alpha_l = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(0.5),             
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp_alpha_m = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(0.5),             
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp_pred = nn.Sequential(
            nn.Linear(embedding_dim * 2, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim), 
            nn.Dropout(0.5),             
            nn.Linear(output_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, history_embeddings, z_l, z_m, z_s, item_embedding):
        # Long-term history feature extraction
        h_l, _ = self.gru_l(history_embeddings)
        h_l = h_l[:, -1, :]  # Get the last hidden state
        h_l = self.bn_l(h_l)  # Apply BatchNorm
        h_l = self.dropout_l(h_l)  # Apply Dropout

        # Mid-term history feature extraction
        h_m, _ = self.gru_m(history_embeddings[:, -k_m:, :])  # Use the most recent k_m embeddings
        h_m = h_m[:, -1, :]  # Get the last hidden state
        h_m = self.bn_m(h_m)  # Apply BatchNorm
        h_m = self.dropout_m(h_m)  # Apply Dropout

        # Calculate attention weights
        alpha_l = self.mlp_alpha_l(torch.cat((h_l, z_l, z_m), dim=1))
        alpha_m = self.mlp_alpha_m(torch.cat((h_m, z_m, z_s), dim=1))

        # Final interest representation
        z_t = alpha_l * z_l + (1 - alpha_l) * alpha_m * z_m + (1 - alpha_l) * (1 - alpha_m) * z_s
        # Predict the likelihood of selecting an item
        y_int = self.mlp_pred(torch.cat((z_t, item_embedding), dim=1))

        return y_int

class BCELossModule(nn.Module):
    def __init__(self):
        super(BCELossModule, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, y_pred, y_true):
        """
        Calculate the Binary Cross-Entropy Loss.
        y_pred: Predicted labels for positive classes.
        y_true: True labels.
        """
        return self.loss_fn(y_pred, y_true)
    
class CAMP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim, output_dim):
        super(CAMP, self).__init__()

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        
        # Interest modules
        self.long_term_module = LongTermInterestModule(embedding_dim)
        self.mid_term_module = MidTermInterestModule(embedding_dim, hidden_dim)
        self.short_term_module = ShortTermInterestModule(embedding_dim, hidden_dim)
        
        # Interest fusion and BCE loss
        self.interest_fusion_module = InterestFusionModule(embedding_dim, hidden_dim, output_dim)
        self.bce_loss_module = BCELossModule()

    def forward(self, user_ids, item_ids, history_items_padded, _mid_len, _short_len, neg_item_ids):
        # Embed user, item, and negative item IDs
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        history_embeds = self.item_embedding(history_items_padded)
        neg_item_embeds = self.item_embedding(neg_item_ids)       

        # Interest calculations using embedded history
        z_l = self.long_term_module(history_embeds, user_embeds)
        z_m = self.mid_term_module(history_embeds, user_embeds)
        z_s = self.short_term_module(history_embeds, user_embeds)

        # Interest proxies and contrastive loss calculation
        p_l = long_term_interest_proxy(history_embeds)
        p_m = mid_term_interest_proxy(history_embeds, _mid_len)        
        p_s = short_term_interest_proxy(history_embeds, _short_len)
        loss_con = calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s)

        # Final interest representation and predictions
        y_int_pos = self.interest_fusion_module(history_embeds, z_l, z_m, z_s, item_embeds)
        y_int_neg = self.interest_fusion_module(history_embeds, z_l, z_m, z_s, neg_item_embeds)

        # BCE Loss calculation
        loss_bce = self.bce_loss_module(y_int_pos, torch.ones_like(y_int_pos)) + \
                   self.bce_loss_module(y_int_neg, torch.zeros_like(y_int_neg))

        # Total loss
        loss = loss_con + loss_bce
        return loss, y_int_pos, y_int_neg

#################################################################### Metrics
def precision_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def ndcg_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_list = predicted[:k]
    dcg = 0.0
    for i, pred in enumerate(pred_list):
        if pred in act_set:
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) because i+1 is 1-based index and +1 for log
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(act_set), k))])
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    return 1.0 if len(act_set & pred_set) > 0 else 0.0

##############################################################################################  data preprocessing, loading
# 1) Dataset preprocessing
def load_dataset(file_path):   
    df = pd.read_csv(file_path)
    df = df.rename(columns={'parent_asin': 'item_id'})          
    return df

def get_history(group):
    history = []
    histories = []
    for item in group:
        histories.append(list(history))
        history.append(item)
    return histories

def calculate_ranges(group, k_m, k_s):
    group['mid_len'] = group['timestamp'].apply(lambda x: ((group['timestamp'] >= x - k_m) & (group['timestamp'] < x)).sum())
    group['short_len'] = group['timestamp'].apply(lambda x: ((group['timestamp'] >= x - k_s) & (group['timestamp'] < x)).sum())
    return group[['mid_len', 'short_len']] 

def encode_history_items(item_mapping, max_length, history):
    if not history:
        return [0] * max_length
    encoded_items = [item_mapping[item] for item in history]
    return [0] * (max_length - len(encoded_items)) + encoded_items

def generate_negative_sample(all_item_ids, row):
    item = row['item_encoded'] 
    history = row['history_encoded']
    non_interacted_items = list(all_item_ids - set(history) - {item})
    result = random.choice(non_interacted_items) if non_interacted_items else None
    return result

def can_items(all_item_ids, row): 
    history = row['history_encoded']
    result = list(all_item_ids - set(history))
    return result

def preprocess_df(df, user_encoder, item_encoder, k_m, k_s):  
    df['history'] = df.groupby('user_id')['item_id'].transform(get_history)  
    ranges_df = df.groupby('user_id', group_keys=False).apply(lambda x: calculate_ranges(x, k_m, k_s), include_groups=False)
    df = pd.concat([df, ranges_df], axis=1) 

    df['user_encoded'] = user_encoder.transform(df['user_id'])
    df['item_encoded'] = item_encoder.transform(df['item_id']) + 1

    item_mapping = dict(zip(item_encoder.classes_, range(1, len(item_encoder.classes_)+1)))    
    df['history_encoded'] = df['history'].apply(lambda x: encode_history_items(item_mapping, max_length, x))

    all_item_ids = set(range(1, len(item_encoder.classes_)+1)) 
    df['neg_item'] = df.apply(lambda x: generate_negative_sample(all_item_ids, x), axis=1)

    # Splitting the DataFrame into train, validation, and test sets
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    train_df = df.groupby('user_id').apply(lambda x: x.iloc[:-2], include_groups=False).reset_index(drop=True)
    valid_df = df.groupby('user_id').apply(lambda x: x.iloc[-2:-1], include_groups=False).reset_index(drop=True)
    test_df = df.groupby('user_id').apply(lambda x: x.iloc[-1:], include_groups=False).reset_index(drop=True)       
         
    valid_df['can_items'] = valid_df.apply(lambda x: can_items(all_item_ids, x), axis=1)
    # test_df['can_items'] = test_df.apply(lambda x: can_items(all_item_ids, x), axis=1)

    print("train_df\n", train_df)
    
    return train_df, valid_df, test_df

class MakeDataset(Dataset):
    def __init__(self, users, items, histories, mid_lens, short_lens, neg_items):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.histories = [torch.tensor(h, dtype=torch.long) for h in histories]
        self.mid_lens = torch.tensor(mid_lens, dtype=torch.int)
        self.short_lens = torch.tensor(short_lens, dtype=torch.int)
        self.neg_items = torch.tensor(neg_items, dtype=torch.long)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.histories[idx], self.mid_lens[idx], self.short_lens[idx],self.neg_items[idx]

def create_datasets(train_df, valid_df, test_df):
    train_dataset = MakeDataset(
        train_df['user_encoded'], train_df['item_encoded'], train_df['history_encoded'],
        train_df['mid_len'], train_df['short_len'], train_df['neg_item']
    )
    valid_dataset = MakeDataset(
        valid_df['user_encoded'], valid_df['item_encoded'], valid_df['history_encoded'],
        valid_df['mid_len'], valid_df['short_len'], valid_df['neg_item']
    )
    test_dataset = MakeDataset(
        test_df['user_encoded'], test_df['item_encoded'], test_df['history_encoded'],
        test_df['mid_len'], test_df['short_len'], test_df['neg_item']
    )
    return train_dataset, valid_dataset, test_dataset


############################################################################ train, valid, test
#  2) Training
def train(model, data_loader, optimizer, device):
    model.train()  
    total_loss = 0
    
    for user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds  in tqdm(data_loader, desc="Training"):
        user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds = \
            user_embeds.to(device), item_embeds.to(device), history_item_embeds.to(device), mid_lens.to(device), short_lens.to(device), neg_item_embeds.to(device)

        optimizer.zero_grad() 
        
        # Forward pass and loss calculation
        loss, _, _ = model(user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds)
        loss = loss.mean()
        total_loss += loss.item()
        
        loss.backward()  

        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()  # Update parameters
        
    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, device, k=10):
    model.eval()  
    total_loss = 0
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_rates = []

    with torch.no_grad():  
        for user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds in tqdm(data_loader, desc="Evaluating"):
            user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds = \
                user_embeds.to(device), item_embeds.to(device), history_item_embeds.to(device), mid_lens.to(device), short_lens.to(device), neg_item_embeds.to(device)

            # Assuming the model outputs the predicted items in sorted order of preference
            loss, y_int, _ = model(user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds)
            
            # loss = torch.mean(loss).item()  # Placeholder for actual loss computation
            loss = loss.mean()
            total_loss += loss.item()
            
            # Convert predicted_items to actual item indices (top-k)
            # _, top_k_indices = torch.topk(predicted_items, k)
            # predicted_items_top_k = top_k_indices.cpu().numpy()

            # Compute metrics
            # for batch_idx in range(user_embeds.size(0)):
            #     true_items = actual_items[batch_idx]
            #     preds = predicted_items_top_k[batch_idx]
            #     precision_scores.append(precision_at_k(true_items, preds, k))
            #     recall_scores.append(recall_at_k(true_items, preds, k))
            #     ndcg_scores.append(ndcg_at_k(true_items, preds, k))
            #     hit_rates.append(hit_rate_at_k(true_items, preds, k))
    
    average_loss = total_loss / len(data_loader)
    # avg_precision = np.mean(precision_scores)
    # avg_recall = np.mean(recall_scores)
    # avg_ndcg = np.mean(ndcg_scores)
    # avg_hit_rate = np.mean(hit_rates)
    
    print(f"Average Validation Loss: {average_loss:.4f}")
    # print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")

    # return average_loss, avg_precision, avg_recall, avg_ndcg, avg_hit_rate
    return average_loss

# 4) Testing
def test(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # No gradients needed
        for user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds in tqdm(data_loader, desc="Testing"):
            user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds = \
            user_embeds.to(device), item_embeds.to(device), history_item_embeds.to(device), mid_lens.to(device), short_lens.to(device), neg_item_embeds.to(device)
            
            # Assuming your model returns loss and some form of predictions
            loss, y_int_pos, y_int_neg = model(user_embeds, item_embeds, history_item_embeds, mid_lens, short_lens, neg_item_embeds)
            loss = loss.mean()
            total_loss += loss.item()

            pos_corrected = torch.ge(y_int_pos, 0.5) 
            correct_predictions += torch.sum(pos_corrected) 
            neg_corrected = torch.lt(y_int_neg, 0.5)  
            correct_predictions += torch.sum(neg_corrected)  
            total_predictions += (y_int_pos.numel() + y_int_neg.numel())
    
    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    print(f"Test Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return average_loss, accuracy

########################################################### config
max_length = 128
time_unit = 60*60*24*1000# a day
k_m = 3*12*30*time_unit # three year
k_s = 6*30*time_unit # 6 month
batch_size = 128

# Instantiate the model
embedding_dim = 128
hidden_dim = 256
output_dim = 1

num_epochs = 10  # Example epoch count

###################################################################### main
logging.basicConfig(filename='../../log.txt', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

dataset_name = 'sampled_Home_and_Kitchen'
df = load_dataset(f'../../dataset/{dataset_name}/{dataset_name}.csv')
len_df = len(df)
user_encoder = LabelEncoder().fit(df['user_id'])
item_encoder = LabelEncoder().fit(df['item_id'])
num_users = df['user_id'].nunique()
num_items = df['item_id'].nunique()

print("Preraing dataset......")
start_time = time.time()
train_df, valid_df, test_df = preprocess_df(df, user_encoder, item_encoder, k_m, k_s)
end_time = time.time()
pre_t = end_time - start_time
print(f"Dataset prepared in {pre_t:.2f} seconds")

print("Data Loading......")
train_dataset, valid_dataset, test_dataset = create_datasets(train_df, valid_df, test_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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

logging.info(f'Number of users: {num_users}, Number of interactions: {len_df}, Dataset preparation time: {pre_t} seconds, DataLoader loading time: {load_t} seconds, Training time: {train_t} seconds\n')

# Evaluate on test set
test_loss, test_accuracy = test(model, test_loader, device)
