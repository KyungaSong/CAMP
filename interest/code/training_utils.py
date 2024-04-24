import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from evaluate import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k

def train(model, data_loader, optimizer, device):
    model.train()  
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Move all tensors in batch to the specified device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad() 
        
        # Forward pass and loss calculation
        loss, _, _ = model(batch)
        loss = loss.mean()
        total_loss += loss.item()
        
        loss.backward()  
        optimizer.step()  # Update parameters
        
    average_loss = total_loss / len(data_loader)
    print(f"Average Training Loss: {average_loss:.4f}")
    return average_loss

# 3) Evaluating
def evaluate(model, data_loader, device):
    model.eval()  
    total_loss = 0 
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Assuming the model outputs the predicted items in sorted order of preference
            loss, y_int_pos, y_int_negs = model(batch)
            loss = loss.mean()
            total_loss += loss.item()

            pos_corrected = torch.ge(y_int_pos, 0.5) 
            correct_predictions += torch.sum(pos_corrected) 
            # Check negative predictions - assuming y_int_negs is of shape [batch_size, num_negatives]
            neg_corrected = torch.lt(y_int_negs, 0.5) 
            correct_predictions += neg_corrected.sum()  # Sum all the True Negatives
            total_predictions += (y_int_pos.numel() + y_int_negs.numel())       
    
    average_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.float() / total_predictions
    
    print(f"Average Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")
    return average_loss

# 4) Testing
def test(model, data_loader, device, k=10):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_top_k_items = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_rates = []
    
    with torch.no_grad():  
        for batch in tqdm(data_loader, desc="Testing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            loss, _, y_int_negs = model(batch)
            loss = loss.mean()
            total_loss += loss.item()

            # Get the top k indices from y_int_negs
            _, top_k_indices = torch.topk(y_int_negs, k, dim=1)
            # Gather the top k negative item IDs
            top_k_neg_item_ids = torch.gather(batch['neg_items'], 1, top_k_indices)
            all_top_k_items.append(top_k_neg_item_ids.cpu().numpy())

            # Calculate metrics
            actual_items = batch['item'].unsqueeze(1)
            hits = (top_k_neg_item_ids == actual_items).any(dim=1).float()  # Check if actual item is in top-k predictions
            
            precision = hits.mean().item()
            recall = hits.mean().item()
            log_positions = torch.log2(top_k_indices.float() + 2.0)
            ndcg_contributions = torch.div(torch.log2(torch.tensor(2.0, device=device)), log_positions) * hits.unsqueeze(1)
            ndcg = ndcg_contributions.sum() / len(batch['user'])
            hit_rate = hits.mean().item()

            precision_scores.append(precision)
            recall_scores.append(recall)
            ndcg_scores.append(ndcg.item())
            hit_rates.append(hit_rate)
    
    average_loss = total_loss / len(data_loader)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)
    avg_hit_rate = np.mean(hit_rates)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}, NDCG@{k}: {avg_ndcg:.4f}, Hit Rate@{k}: {avg_hit_rate:.4f}")
    return average_loss, all_top_k_items, avg_precision, avg_recall, avg_ndcg, avg_hit_rate