import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # nn.BatchNorm1d(hidden_dim), 
            nn.Dropout(0.5),             
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp_alpha_m = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim), 
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
    
    def forward(self, history_embeddings, mid_lens, z_l, z_m, z_s, item_embedding):
        # Long-term history feature extraction
        h_l, _ = self.gru_l(history_embeddings)
        h_l = h_l[:, -1, :]  # Get the last hidden state
        h_l = self.bn_l(h_l)
        h_l = self.dropout_l(h_l)  

        # Mid-term history feature extraction
        batch_size, seq_len, _ = history_embeddings.size()     
        device = history_embeddings.device 
        masks = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= (seq_len - mid_lens.unsqueeze(1).to(device))  
        masks = masks.to(history_embeddings.device)
        
        # 마스킹 적용
        masked_embeddings = history_embeddings * masks.unsqueeze(-1).float()

        # GRU 실행
        h_m, _ = self.gru_m(masked_embeddings) # Use the most recent mid_lens embeddings
        h_m = h_m[:, -1, :]  # Get the last hidden state
        h_m = self.bn_l(h_m)
        h_m = self.dropout_m(h_m)  

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

    def forward(self, user_ids, item_ids, history_items_padded, mid_lens, short_lens, neg_item_ids):
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
        p_m = mid_term_interest_proxy(history_embeds, mid_lens)        
        p_s = short_term_interest_proxy(history_embeds, short_lens)
        loss_con = calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s)

        # Final interest representation and predictions
        y_int_pos = self.interest_fusion_module(history_embeds, mid_lens, z_l, z_m, z_s, item_embeds)
        y_int_neg = self.interest_fusion_module(history_embeds, mid_lens, z_l, z_m, z_s, neg_item_embeds)

        # BCE Loss calculation
        loss_bce = self.bce_loss_module(y_int_pos, torch.ones_like(y_int_pos)) + \
                   self.bce_loss_module(y_int_neg, torch.zeros_like(y_int_neg))

        # Total loss
        loss = loss_con + loss_bce
        return loss, y_int_pos, y_int_neg