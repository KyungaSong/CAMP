import torch
import torch.nn as nn
import torch.nn.functional as F

class LongTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super(LongTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.W_l = nn.Parameter(torch.Tensor(self.combined_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_l) 
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_dim, 1)            
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, 2 * embedding_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)
    
    def forward(self, item_his_embeds, cat_his_embeds, user_embed):        
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)  # (batch_size, seq_len, combined_dim)
        h = torch.matmul(combined_embeds, self.W_l)  # (batch_size, seq_len, combined_dim)
        user_embed_transformed = self.user_transform(user_embed)  # (batch_size, combined_dim)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)  # (batch_size, 1, combined_dim)
        
        diff = h - user_embed_expanded  # (batch_size, seq_len, combined_dim)
        prod = h * user_embed_expanded  # (batch_size, seq_len, combined_dim)
        last_hidden_nn_layer = torch.cat((h, user_embed_expanded.expand_as(h), diff, prod), dim=-1)  # (batch_size, seq_len, 4 * combined_dim)
        
        alpha = self.mlp(last_hidden_nn_layer).squeeze(2)  # (batch_size, seq_len)
        a = torch.softmax(alpha, dim=1)  # (batch_size, seq_len)
        z_l = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)  # (batch_size, combined_dim)
        return z_l

class MidTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(MidTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.rnn = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.W_m = nn.Parameter(torch.Tensor(hidden_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_m)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, hidden_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, item_his_embeds, cat_his_embeds, user_embed):
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)  # (batch_size, seq_len, combined_dim)
        o, _ = self.rnn(combined_embeds)  # (batch_size, seq_len, hidden_dim)
        h = torch.matmul(o, self.W_m)  # (batch_size, seq_len, combined_dim)
        user_embed_transformed = self.user_transform(user_embed)  # (batch_size, hidden_dim)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        diff = h - user_embed_expanded  # (batch_size, seq_len, combined_dim)
        prod = h * user_embed_expanded  # (batch_size, seq_len, combined_dim)
        last_hidden_nn_layer = torch.cat((h, user_embed_expanded.expand_as(h), diff, prod), dim=-1)  # (batch_size, seq_len, 4 * combined_dim)
        
        alpha = self.mlp(last_hidden_nn_layer).squeeze(2)  # (batch_size, seq_len)
        a = torch.softmax(alpha, dim=1)  # (batch_size, seq_len)
        z_m = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)  # (batch_size, combined_dim)
        return z_m

class ShortTermInterestModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(ShortTermInterestModule, self).__init__()
        self.combined_dim = 2 * embedding_dim 
        self.rnn = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.W_s = nn.Parameter(torch.Tensor(hidden_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_s)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, hidden_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)

        # Initialize GRU weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, item_his_embeds, cat_his_embeds, user_embed):
        combined_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)  # (batch_size, seq_len, combined_dim)
        o, _ = self.rnn(combined_embeds)  # (batch_size, seq_len, hidden_dim)
        h = torch.matmul(o, self.W_s)  # (batch_size, seq_len, combined_dim)
        user_embed_transformed = self.user_transform(user_embed)  # (batch_size, hidden_dim)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        diff = h - user_embed_expanded  # (batch_size, seq_len, combined_dim)
        prod = h * user_embed_expanded  # (batch_size, seq_len, combined_dim)
        last_hidden_nn_layer = torch.cat((h, user_embed_expanded.expand_as(h), diff, prod), dim=-1)  # (batch_size, seq_len, 4 * combined_dim)
        
        alpha = self.mlp(last_hidden_nn_layer).squeeze(2)  # (batch_size, seq_len)
        a = torch.softmax(alpha, dim=1)  # (batch_size, seq_len)
        z_s = torch.sum(a.unsqueeze(2) * combined_embeds, dim=1)  # (batch_size, combined_dim)
        return z_s
    
def long_term_interest_proxy(item_his_embeds, cat_his_embeds):
    """
    Calculate the long-term interest proxy using both item and category embeddings.
    """
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    p_l_t = torch.mean(combined_history_embeds, dim=1)
    return p_l_t

def mid_term_interest_proxy(item_his_embeds, cat_his_embeds, mid_lens):
    """
    Calculate the mid-term interest proxy using masking for variable lengths and both item and category embeddings.
    """
    device = item_his_embeds.device
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    max_len = combined_history_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(mid_lens), max_len) < mid_lens.unsqueeze(1).to(device)
    
    masked_history = combined_history_embeds * mask.unsqueeze(-1).type_as(combined_history_embeds)
    valid_counts = mask.sum(1, keepdim=True)
    
    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_m_t = masked_history.sum(1) / safe_valid_counts.type_as(combined_history_embeds)
    p_m_t = torch.nan_to_num(p_m_t, nan=0.0)

    return p_m_t

def short_term_interest_proxy(item_his_embeds, cat_his_embeds, short_lens):
    """
    Calculate the short-term interest proxy using both item and category embeddings.
    """
    device = item_his_embeds.device
    combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
    max_len = combined_history_embeds.size(1)
    mask = torch.arange(max_len, device=device).expand(len(short_lens), max_len) < short_lens.unsqueeze(1).to(device)
    
    masked_history = combined_history_embeds * mask.unsqueeze(-1).type_as(combined_history_embeds)
    valid_counts = mask.sum(1, keepdim=True)
    
    safe_valid_counts = torch.where(valid_counts > 0, valid_counts, torch.ones_like(valid_counts))
    p_s_t = masked_history.sum(1) / safe_valid_counts.type_as(combined_history_embeds)
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

def calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s, has_mid):
    """
    Calculate the overall contrastive loss L_con for a user at time t,
    which is the sum of L_lm (long-mid term contrastive loss) and L_ms (mid-short term contrastive loss).
    """
    # Loss for the long-term and mid-term interests pair
    L_lm = bpr_loss(z_l, p_l, p_m) + bpr_loss(p_l, z_l, z_m) + \
           bpr_loss(z_m, p_m, p_l) + bpr_loss(p_m, z_m, z_l)
    
    if has_mid:
    # Loss for the mid-term and short-term interests pair
        L_ms = bpr_loss(z_m, p_m, p_s) + bpr_loss(p_m, z_m, z_s) + \
            bpr_loss(z_s, p_s, p_m) + bpr_loss(p_s, z_s, z_m)
        L_con = L_lm + L_ms
    else:
        L_con = L_lm 
    return L_con

def compute_discrepancy_loss(a, b, discrepancy_loss_weight):
    """
    Calculate the discrepancy loss between two embeddings a and b.
    """
    discrepancy_loss = F.mse_loss(a, b)
    discrepancy_loss = -discrepancy_loss_weight * discrepancy_loss
    return discrepancy_loss

class InterestFusionModule(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, dropout_rate):
        super(InterestFusionModule, self).__init__()
        self.combined_dim = 2 * embedding_dim  
        self.gru_l = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.gru_m = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        input_dim_for_alpha = 3 * hidden_dim 

        self.mlp_alpha_l = nn.Sequential(
            nn.Linear(input_dim_for_alpha, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.mlp_alpha_m = nn.Sequential(
            nn.Linear(input_dim_for_alpha, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.mlp_pred = nn.Sequential(
            nn.Linear(embedding_dim * 4, output_dim),  
            nn.ReLU(),            
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, item_embeds, cat_embeds, has_mid):
        # Combine item and category history embeddings
        combined_history_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        
        # Long-term history feature extraction
        h_l, _ = self.gru_l(combined_history_embeds)
        h_l = h_l[:, -1, :]  

        # Mid-term history feature extraction
        if has_mid:
            batch_size, seq_len, _ = combined_history_embeds.size()     
            masks = torch.arange(seq_len, device=mid_lens.device).expand(batch_size, seq_len) >= (seq_len - mid_lens.unsqueeze(1))   
            masked_embeddings = combined_history_embeds * masks.unsqueeze(-1).float()

            h_m, _ = self.gru_m(masked_embeddings)
            h_m = h_m[:, -1, :] 

        # Attention weights
        alpha_l = self.mlp_alpha_l(torch.cat((h_l, z_l, z_m), dim=1))
        if has_mid:
            alpha_m = self.mlp_alpha_m(torch.cat((h_m, z_m, z_s), dim=1))
            z_t = alpha_l * z_l + (1 - alpha_l) * alpha_m * z_m + (1 - alpha_l) * (1 - alpha_m) * z_s
        else:
            z_t = alpha_l * z_l + (1 - alpha_l) * z_m 
        y_int = self.mlp_pred(torch.cat((z_t, item_embeds, cat_embeds), dim=1))

        return y_int

class BCELossModule(nn.Module):
    def __init__(self, pos_weight):
        super(BCELossModule, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class CAMP(nn.Module):
    def __init__(self, num_users, num_items, num_cats, config):
        super(CAMP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats + 1, config.embedding_dim, padding_idx=0)
        self.long_term_module = LongTermInterestModule(config.embedding_dim, config.dropout_rate)
        self.mid_term_module = MidTermInterestModule(config.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.short_term_module = ShortTermInterestModule(config.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.interest_fusion_module = InterestFusionModule(config.embedding_dim, config.hidden_dim, config.output_dim, config.dropout_rate)
        self.bce_loss_module = BCELossModule(pos_weight=torch.tensor([4.0]))
        self.regularization_weight = config.regularization_weight
        self.discrepancy_loss_weight = config.discrepancy_loss_weight
        self.has_mid = config.has_mid

    def forward(self, batch, item_to_cat_dict, device):
        user_ids = batch['user']
        item_ids = batch['item']
        cat_ids = batch['cat']
        items_history_padded = batch['item_his']        
        cats_history_padded = batch['cat_his']  
        if self.has_mid:
            mid_lens = batch['mid_len']   
        else:
            mid_lens = batch['short_len'] 
        short_lens = batch['short_len']
        neg_items_ids = batch['neg_items']
        neg_cats_ids = torch.tensor([[item_to_cat_dict[item.item()] for item in neg_items] for neg_items in neg_items_ids], device=device)

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        cat_embeds = self.cat_embedding(cat_ids)
        item_his_embeds = self.item_embedding(items_history_padded)
        cat_his_embeds = self.cat_embedding(cats_history_padded)
        neg_items_embeds = self.item_embedding(neg_items_ids)
        neg_cats_embeds = self.cat_embedding(neg_cats_ids)

        z_l = self.long_term_module(item_his_embeds, cat_his_embeds, user_embeds)
        z_m = self.mid_term_module(item_his_embeds, cat_his_embeds, user_embeds)
        z_s = self.short_term_module(item_his_embeds, cat_his_embeds, user_embeds)

        p_l = long_term_interest_proxy(item_his_embeds, cat_his_embeds)
        p_m = mid_term_interest_proxy(item_his_embeds, cat_his_embeds, mid_lens)
        p_s = short_term_interest_proxy(item_his_embeds, cat_his_embeds, short_lens)
        loss_con = calculate_contrastive_loss(z_l, z_m, z_s, p_l, p_m, p_s, self.has_mid)

        y_int_pos = self.interest_fusion_module(item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, item_embeds, cat_embeds, self.has_mid)     
        y_int_negs = torch.stack([
            self.interest_fusion_module(item_his_embeds, cat_his_embeds, mid_lens, z_l, z_m, z_s, neg_embed.squeeze(1), cat_embed.squeeze(1), self.has_mid)
            for neg_embed, cat_embed in zip(neg_items_embeds.split(1, dim=1), neg_cats_embeds.split(1, dim=1))
        ], dim=1).squeeze(2) 

        loss_bce_pos = self.bce_loss_module(y_int_pos, torch.ones_like(y_int_pos))
        loss_bce_neg = self.bce_loss_module(y_int_negs, torch.zeros_like(y_int_negs))
        loss_bce = loss_bce_pos + loss_bce_neg

        loss_discrepancy = compute_discrepancy_loss(z_l, z_m, self.discrepancy_loss_weight)
        if self.has_mid:
            loss_discrepancy_ms = compute_discrepancy_loss(z_m, z_s, self.discrepancy_loss_weight)
            loss_discrepancy += loss_discrepancy_ms

        # Regularization loss (L2 loss)
        regularization_loss = self.regularization_weight * sum(torch.norm(param) for param in self.parameters())

        loss = loss_con + loss_bce + loss_discrepancy + regularization_loss
        return loss, y_int_pos, y_int_negs
