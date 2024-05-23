import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim

from evaluate import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k
from sklearn.metrics import roc_auc_score

def train(model, data_loader, optimizer, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device, rank):
    model.train()
    total_loss = 0

    if rank == 0:
        pbar = tqdm(data_loader, desc="Training")
    else:
        pbar = data_loader

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        loss, y_int_pos, y_int_negs = model(batch, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device)
        # print(f"y_int_pos: {y_int_pos[0]}\n, y_int_negs: {y_int_negs[0]}\n, loss", {loss[0]})
        loss = loss.mean()        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device, rank):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    if rank == 0:
        pbar = tqdm(data_loader, desc="Evaluating")
    else:
        pbar = data_loader

    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, y_int_pos, y_int_negs = model(batch, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device)
            loss = loss.mean()
            total_loss += loss.item()

            pos_corrected = torch.ge(y_int_pos, 0.5)
            correct_predictions += pos_corrected.sum().item()  
            neg_corrected = torch.lt(y_int_negs, 0.5)
            correct_predictions += neg_corrected.sum().item()  
            total_predictions += y_int_pos.numel() + y_int_negs.numel()

    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions 
    print(f"Average Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return average_loss, accuracy

# 4) Testing
def test(model, data_loader, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device, rank, k=10):
    model.eval()  
    total_loss = 0
    all_top_k_items = []
    all_top_2_items = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    ndcg_2_scores = []
    hit_rates = []
    auc_scores = []
    mrr_scores = []
    
    if rank == 0:
        pbar = tqdm(data_loader, desc="Testing")
    else:
        pbar = data_loader

    with torch.no_grad():
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, y_int_pos, y_int_negs = model(batch, item_to_cat_dict, item_to_con_dict, item_to_qlt_dict, device)
            loss = loss.mean()
            total_loss += loss.item()

            _, top_k_indices = torch.topk(y_int_negs, k, dim=1)
            top_k_neg_item_ids = torch.gather(batch['neg_items'], 1, top_k_indices)
            all_top_k_items.append(top_k_neg_item_ids.cpu().numpy())
            
            _, top_2_indices = torch.topk(y_int_negs, 2, dim=1)
            top_2_neg_item_ids = torch.gather(batch['neg_items'], 1, top_2_indices)
            all_top_2_items.append(top_2_neg_item_ids.cpu().numpy())

            actual_items = batch['item'].unsqueeze(1)
            hits = (top_k_neg_item_ids == actual_items).any(dim=1).float()
            hits_2 = (top_2_neg_item_ids == actual_items).any(dim=1).float()

            precision = (hits.sum() / len(batch['user'])).item() 
            recall = (hits.sum() / len(batch['user'])).item()
            hit_rate = hits.mean().item()

            # NDCG@k
            actual_items_expand = actual_items.expand_as(top_k_neg_item_ids)
            dcg = torch.sum((top_k_neg_item_ids == actual_items_expand).float() / torch.log2(top_k_indices.float() + 2), dim=1)
            idcg = torch.sum(1.0 / torch.log2(torch.arange(1, k + 1).float() + 1).to(device))
            ndcg = (dcg / idcg).mean().item()

            # NDCG@2
            actual_items_expand_2 = actual_items.expand_as(top_2_neg_item_ids)
            dcg_2 = torch.sum((top_2_neg_item_ids == actual_items_expand_2).float() / torch.log2(top_2_indices.float() + 2), dim=1)
            idcg_2 = torch.sum(1.0 / torch.log2(torch.arange(1, 3).float() + 1).to(device))
            ndcg_2 = (dcg_2 / idcg_2).mean().item()

            true_labels = torch.cat((torch.ones_like(y_int_pos), torch.zeros_like(y_int_negs)), dim=1).flatten()
            predictions = torch.cat((y_int_pos, y_int_negs), dim=1).flatten()
            auc_score = roc_auc_score(true_labels.cpu(), predictions.cpu())
            auc_scores.append(auc_score)

            correct_ranks = torch.where(top_k_neg_item_ids == actual_items)[1]
            mrr = torch.reciprocal((correct_ranks.float() + 1)).mean().item()
            mrr_scores.append(mrr)

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg)
            ndcg_2_scores.append(ndcg_2)
            hit_rates.append(hit_rate)
    
    average_loss = total_loss / len(data_loader)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)
    avg_ndcg_2 = np.mean(ndcg_2_scores)
    avg_hit_rate = np.mean(hit_rates)
    avg_auc = np.mean(auc_scores)
    avg_mrr = np.mean(mrr_scores)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, NDCG@2: {avg_ndcg_2:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")
    print(f"AUC: {avg_auc:.4f}, MRR: {avg_mrr:.4f}")
    return average_loss, all_top_k_items, avg_precision, avg_recall, avg_ndcg, avg_ndcg_2, avg_hit_rate, avg_auc, avg_mrr

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
