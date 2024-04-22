import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from evaluate import precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k

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

        # clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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