import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from evaluate import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k
from sklearn.metrics import roc_auc_score

def train(model, data_loader, optimizer, item_to_cat_dict, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        loss, _, _ = model(batch, item_to_cat_dict, device)
        loss = loss.mean()  
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, item_to_cat_dict, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, y_int_pos, y_int_negs = model(batch, item_to_cat_dict, device)
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
def test(model, data_loader, item_to_cat_dict, device, k=10):
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
    
    with torch.no_grad():  
        for batch in tqdm(data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, y_int_pos, y_int_negs = model(batch, item_to_cat_dict, device)
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

            precision = (hits.sum() / (k * len(batch['user']))).item() 
            recall = (hits.sum() / len(batch['user'])).item()
            log_positions = torch.log2(top_k_indices.float() + 2.0)
            ndcg_contributions = torch.div(torch.log2(torch.tensor(2.0)), log_positions) * hits.unsqueeze(1)
            ndcg = (ndcg_contributions.sum() / len(batch['user'])).item()
            ndcg_2_contributions = torch.div(torch.log2(torch.tensor(2.0)), log_positions) * hits_2.unsqueeze(1)
            ndcg_2 = (ndcg_2_contributions.sum() / len(batch['user'])).item()
            hit_rate = hits.mean().item()

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
