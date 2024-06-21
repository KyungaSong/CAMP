import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import roc_auc_score

def train(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        loss, _ = model(batch, device)
        loss = loss.mean()        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, _ = model(batch, device)
            loss = loss.mean()
            total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Validation Loss: {average_loss:.4f}")
    return average_loss

# 4) Testing
# def test(model, data_loader, device, rank, k=10):
#     model.eval()  
#     total_loss = 0
#     precision_scores = []
#     recall_scores = []
#     ndcg_scores = []
#     hit_rates = []
#     auc_scores = []
#     mrr_scores = []

#     if rank == 0:
#         pbar = tqdm(data_loader, desc="Testing")
#     else:
#         pbar = data_loader

#     with torch.no_grad():
#         for batch in pbar:
#             batch = {k: v.to(device) for k, v in batch.items()}
            
#             loss, y_int = model(batch, device)
#             loss = loss.mean()
#             total_loss += loss.item()

#             # Reshape y_int to include positive and negative samples
#             y_int = y_int.view(-1, 100)  # 1 positive + 99 negative samples

#             # Calculate top-k items
#             _, top_k_indices = torch.topk(y_int, k, dim=1)
#             top_k_items = torch.gather(batch['item'].view(-1, 100), 1, top_k_indices)

#             actual_items = batch['item'].view(-1, 100)[:, 0].unsqueeze(1)
#             hits = (top_k_items == actual_items).any(dim=1).float()

#             precision = (hits.sum() / k).item()
#             recall = (hits.sum() / len(actual_items)).item()  # multiple positive samples per user
#             hit_rate = hits.mean().item()

#             # NDCG@k
#             actual_items_expand = actual_items.expand_as(top_k_items)
#             dcg = torch.sum((top_k_items == actual_items_expand).float() / torch.log2(top_k_indices.float() + 2), dim=1)
#             idcg = torch.sum(1.0 / torch.log2(torch.arange(1, k + 1).float() + 1).to(device))
#             ndcg = (dcg / idcg).mean().item()

#             # AUC
#             true_labels = torch.cat([torch.ones_like(y_int[:, :1]), torch.zeros_like(y_int[:, 1:])], dim=1).flatten()
#             predictions = y_int.flatten()
#             auc_score = roc_auc_score(true_labels.cpu(), predictions.cpu())
#             auc_scores.append(auc_score)

#             # MRR
#             ranks = torch.where(top_k_items == actual_items)[1].float() + 1.0
#             mrr = (1.0 / ranks).mean().item()
#             mrr_scores.append(mrr)

#             precision_scores.append(precision)
#             recall_scores.append(recall)
#             ndcg_scores.append(ndcg)
#             hit_rates.append(hit_rate)
    
#     average_loss = total_loss / len(data_loader)
#     avg_precision = np.mean(precision_scores)
#     avg_recall = np.mean(recall_scores)
#     avg_ndcg = np.mean(ndcg_scores)
#     avg_hit_rate = np.mean(hit_rates)
#     avg_auc = np.mean(auc_scores)
#     avg_mrr = np.mean(mrr_scores)

#     print(f"Test Loss: {average_loss:.4f}")
#     print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")
#     print(f"AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")
#     return average_loss, avg_precision, avg_recall, avg_ndcg, avg_hit_rate, avg_auc, avg_mrr

def test(model, data_loader, device, inv, k_list=[5, 10, 20]):
    model.eval()  
    total_loss = 0
    metrics = {k: {'precision_scores': [], 'recall_scores': [], 'ndcg_scores': [], 'hit_rates': [], 'auc_scores': [], 'mrr_scores': []} for k in k_list}

    all_predictions = []
    all_labels = []
    all_user_ids = []
    all_item_ids = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}

            batch['con_his'] *= inv
                
            loss, y_int = model(batch, device)
            loss = loss.mean()
            total_loss += loss.item()

            user_ids = batch['user']
            y_int = y_int.view(-1)  # Flatten y_int for easier processing

            # Collect all predictions, labels, user_ids and item_ids
            all_predictions.extend(y_int.cpu().numpy())
            all_labels.extend(batch['label'].cpu().numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
            all_item_ids.extend(batch['item'].cpu().numpy())

    # Group predictions, labels, and items by user
    
    predictions_by_user = defaultdict(list)
    labels_by_user = defaultdict(list)
    items_by_user = defaultdict(list)

    for pred, label, user_id, item_id in zip(all_predictions, all_labels, all_user_ids, all_item_ids):
        predictions_by_user[user_id].append(pred)
        labels_by_user[user_id].append(label)
        items_by_user[user_id].append(item_id)

    # Calculate metrics per user
    for user_id in predictions_by_user.keys():
        user_predictions = predictions_by_user[user_id]
        user_labels = labels_by_user[user_id]
        user_items = items_by_user[user_id]

        # Sort by prediction score
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

            metrics[k]['precision_scores'].append(precision)
            metrics[k]['recall_scores'].append(recall)
            metrics[k]['ndcg_scores'].append(ndcg)
            metrics[k]['hit_rates'].append(hit_rate)
            metrics[k]['auc_scores'].append(auc_score)
            metrics[k]['mrr_scores'].append(mrr)

    average_loss = total_loss / len(data_loader)
    results = {}
    for k in k_list:
        avg_precision = np.mean(metrics[k]['precision_scores'])
        avg_recall = np.mean(metrics[k]['recall_scores'])
        avg_ndcg = np.mean(metrics[k]['ndcg_scores'])
        avg_hit_rate = np.mean(metrics[k]['hit_rates'])
        avg_auc = np.mean(metrics[k]['auc_scores'])
        avg_mrr = np.mean(metrics[k]['mrr_scores'])

        results[k] = {
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
    
    return average_loss, results

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
