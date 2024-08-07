import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp

# 1) Training
def train(model, data_loader, optimizer, device, config):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        if config.method == 'DICE':
            loss, _, _ = model(batch, device) 
        elif config.method == 'MACR':
            loss, _, _, _, _ = model(batch, device)
        else:
            loss, _ = model(batch, device)

        loss = loss.mean()        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")

    # Adapt weights after each epoch
    if config.method == 'DICE':
        model.adapt(epoch=0, decay=config.loss_decay) 
    return average_loss

# 2) Evaluating
def evaluate(model, data_loader, device, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            if config.method == 'DICE':
                loss, _, _ = model(batch, device)
            elif config.method == 'MACR':
                loss, _, _, _, _ = model(batch, device)
            else:
                loss, _ = model(batch, device)
                
            loss = loss.mean()
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Validation Loss: {average_loss:.4f}")
    return average_loss

# 3) Testing
def calculate_metrics_for_user(user_predictions, user_labels, user_items, k_list):
    user_metrics = {k: {} for k in k_list}
    sorted_indices = np.argsort(user_predictions)[::-1]

    for k in k_list:
        top_k_items = [user_items[i] for i in sorted_indices[:k]]
        actual_items = [user_items[i] for i in range(len(user_labels)) if user_labels[i] == 1]

        hits = [item in actual_items for item in top_k_items]

        precision = sum(hits) / k
        recall = sum(hits) / len(actual_items) if actual_items else 0
        hit_rate = 1 if sum(hits) > 0 else 0

        # NDCG@k
        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(len(actual_items), k)))
        ndcg = dcg / idcg if idcg > 0 else 0

        # AUC
        y_true = user_labels
        y_scores = user_predictions
        auc_score = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0

        # MRR
        ranks = [idx + 1 for idx, hit in enumerate(hits) if hit]
        mrr = (1.0 / ranks[0]) if ranks else 0

        user_metrics[k] = {
            'precision': precision,
            'recall': recall,
            'ndcg': ndcg,
            'hit_rate': hit_rate,
            'auc': auc_score,
            'mrr': mrr
        }
    return user_metrics

def test(model, data_loader, device, config, inv_ratio, k_list=[5, 10, 20]):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    model.set_testing_mode(True)
    
    total_loss = 0
    metrics = {k: {'precision_scores': [], 'recall_scores': [], 'ndcg_scores': [], 'hit_rates': [], 'auc_scores': [], 'mrr_scores': []} for k in k_list}

    all_predictions = []
    all_labels = []
    all_user_ids = []
    all_item_ids = []

    with torch.no_grad():
        if config.method == 'MACR':
            mean_user_embed = model.user_embedding.weight.mean(dim=0, keepdim=True)
            mean_item_embed = model.item_embedding.weight.mean(dim=0, keepdim=True)
            mean_cat_embed = model.cat_embedding.weight.mean(dim=0, keepdim=True)

            mean_combined_his_embeds = torch.cat((mean_item_embed, mean_cat_embed), dim=-1).unsqueeze(0)  
            mean_z_l = model.long_term_module(mean_combined_his_embeds, mean_user_embed)
            mean_z_s = model.short_term_module(mean_combined_his_embeds, mean_user_embed)
            c = model.interest_fusion_module(mean_combined_his_embeds, mean_z_l, mean_z_s, mean_item_embed, mean_cat_embed)

        for batch in tqdm(data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            if config.method == 'CAMP':
                batch['con_his'] *= inv_ratio
                loss, y_pred = model(batch, device)
            elif config.method == 'MACR':
                loss, y_pred, y_k, y_i, y_u = model(batch, device)
                y_pred = y_pred - c * torch.sigmoid(y_i) * torch.sigmoid(y_u)
            else:
                loss, y_pred = model(batch, device)

            loss = loss.mean()
            total_loss += loss.item()

            user_ids = batch['user']
            y_pred = y_pred.view(-1)  

            all_predictions.extend(y_pred.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
            all_item_ids.extend(batch['item'].cpu().numpy())
    
    predictions_by_user = defaultdict(list)
    labels_by_user = defaultdict(list)
    items_by_user = defaultdict(list)

    for pred, label, user_id, item_id in zip(all_predictions, all_labels, all_user_ids, all_item_ids):
        predictions_by_user[user_id].append(pred)
        labels_by_user[user_id].append(label)
        items_by_user[user_id].append(item_id)

    user_ids = list(predictions_by_user.keys())

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(calculate_metrics_for_user, 
                               [(predictions_by_user[user_id], 
                                 labels_by_user[user_id], 
                                 items_by_user[user_id], 
                                 k_list) for user_id in user_ids])

    for result in results:
        for k in k_list:
            metrics[k]['precision_scores'].append(result[k]['precision'])
            metrics[k]['recall_scores'].append(result[k]['recall'])
            metrics[k]['ndcg_scores'].append(result[k]['ndcg'])
            metrics[k]['hit_rates'].append(result[k]['hit_rate'])
            metrics[k]['auc_scores'].append(result[k]['auc'])
            metrics[k]['mrr_scores'].append(result[k]['mrr'])

    average_loss = total_loss / len(data_loader)
    final_results = {}
    for k in k_list:
        avg_precision = np.mean(metrics[k]['precision_scores'])
        avg_recall = np.mean(metrics[k]['recall_scores'])
        avg_ndcg = np.mean(metrics[k]['ndcg_scores'])
        avg_hit_rate = np.mean(metrics[k]['hit_rates'])
        avg_auc = np.mean(metrics[k]['auc_scores'])
        avg_mrr = np.mean(metrics[k]['mrr_scores'])

        final_results[k] = {
            'Precision': avg_precision,
            'Recall': avg_recall,
            'NDCG': avg_ndcg,
            'Hit Rate': avg_hit_rate,
            'AUC': avg_auc,
            'MRR': avg_mrr
        }
        print(f"Metrics for k={k}:")
        print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")
        print(f"AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")

    return average_loss, final_results

def test_DICE(model, data_loader, device, config, k_list=[5, 10, 20]):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for testing")
        model = nn.DataParallel(model)

    model.to(device)
    model.eval()
    model.set_testing_mode(True)
    total_loss = 0
    metrics = {k: {'precision_scores': [], 'recall_scores': [], 'ndcg_scores': [], 'hit_rates': [], 'auc_scores': [], 'mrr_scores': []} for k in k_list}

    all_predictions = []
    all_labels = []
    all_user_ids = []
    all_item_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing DICE"):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, pos_y_pred, neg_y_pred = model(batch, device)
            loss = loss.mean()
            total_loss += loss.item()

            pos_item = batch['pos_item'][0].unsqueeze(0)
            neg_item = batch['neg_item'].view(-1) 
            items = torch.cat([pos_item, neg_item], dim=0) 
            
            pos_y_pred = pos_y_pred[0].unsqueeze(0)             
            y_pred = torch.cat([pos_y_pred, neg_y_pred], dim=0)
            
            labels = torch.cat([torch.ones_like(pos_y_pred), torch.zeros_like(neg_y_pred)], dim=0)  # (batch_size, 1 + test_num_samples)
            all_predictions.extend(y_pred.cpu().numpy().reshape(-1))
            all_labels.extend(labels.cpu().numpy().reshape(-1))
            all_user_ids.extend(batch['user'].cpu().numpy().repeat(1 + config.test_num_samples))
            all_item_ids.extend(items.cpu().numpy().reshape(-1))

    predictions_by_user = defaultdict(list)
    labels_by_user = defaultdict(list)
    items_by_user = defaultdict(list)

    for pred, label, user_id, item_id in zip(all_predictions, all_labels, all_user_ids, all_item_ids):
        predictions_by_user[user_id].append(pred)
        labels_by_user[user_id].append(label)
        items_by_user[user_id].append(item_id)

    user_ids = list(predictions_by_user.keys())

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(calculate_metrics_for_user, 
                               [(predictions_by_user[user_id], 
                                 labels_by_user[user_id], 
                                 items_by_user[user_id], 
                                 k_list) for user_id in user_ids])

    for result in results:
        for k in k_list:
            metrics[k]['precision_scores'].append(result[k]['precision'])
            metrics[k]['recall_scores'].append(result[k]['recall'])
            metrics[k]['ndcg_scores'].append(result[k]['ndcg'])
            metrics[k]['hit_rates'].append(result[k]['hit_rate'])
            metrics[k]['auc_scores'].append(result[k]['auc'])
            metrics[k]['mrr_scores'].append(result[k]['mrr'])

    average_loss = total_loss / len(data_loader)
    final_results = {}
    for k in k_list:
        avg_precision = np.mean(metrics[k]['precision_scores'])
        avg_recall = np.mean(metrics[k]['recall_scores'])
        avg_ndcg = np.mean(metrics[k]['ndcg_scores'])
        avg_hit_rate = np.mean(metrics[k]['hit_rates'])
        avg_auc = np.mean(metrics[k]['auc_scores'])
        avg_mrr = np.mean(metrics[k]['mrr_scores'])

        final_results[k] = {
            'Precision': avg_precision,
            'Recall': avg_recall,
            'NDCG': avg_ndcg,
            'Hit Rate': avg_hit_rate,
            'AUC': avg_auc,
            'MRR': avg_mrr
        }
        print(f"Metrics for k={k}:")
        print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")
        print(f"AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")

    return average_loss, final_results

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
