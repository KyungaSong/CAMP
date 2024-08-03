import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class LongTermInterestModule(nn.Module):
    def __init__(self, combined_dim, embedding_dim, dropout_rate):
        super(LongTermInterestModule, self).__init__()
        self.combined_dim = combined_dim
        self.W_l = nn.Parameter(torch.Tensor(self.combined_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_l)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.combined_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, self.combined_dim)
        nn.init.xavier_uniform_(self.user_transform.weight)
        self.user_bn = nn.BatchNorm1d(self.combined_dim)

    def forward(self, combined_his_embeds, user_embed):
        h = torch.matmul(combined_his_embeds, self.W_l)  # (batch_size, seq_len, combined_dim)
        user_embed_transformed = self.user_transform(user_embed)  # (batch_size, combined_dim)
        user_embed_transformed = self.user_bn(user_embed_transformed)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)  # (batch_size, 1, combined_dim)

        diff = h - user_embed_expanded  # (batch_size, seq_len, combined_dim)
        prod = h * user_embed_expanded  # (batch_size, seq_len, combined_dim)
        last_hidden_nn_layer = torch.cat((h, user_embed_expanded.expand_as(h), diff, prod), dim=-1)  # (batch_size, seq_len, 4 * combined_dim)

        last_hidden_nn_layer_reshaped = last_hidden_nn_layer.view(-1, last_hidden_nn_layer.size(-1))
        alpha = self.mlp(last_hidden_nn_layer_reshaped).squeeze(1)  # (batch_size * seq_len)
        alpha = alpha.view(combined_his_embeds.size(0), combined_his_embeds.size(1))  # (batch_size, seq_len)

        a = torch.softmax(alpha, dim=1)  # (batch_size, seq_len)
        z_l = torch.sum(a.unsqueeze(2) * combined_his_embeds, dim=1)  # (batch_size, combined_dim)
        return z_l

class ShortTermInterestModule(nn.Module):
    def __init__(self, combined_dim, embedding_dim, hidden_dim, dropout_rate):
        super(ShortTermInterestModule, self).__init__()
        self.combined_dim = combined_dim
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        self.W_s = nn.Parameter(torch.Tensor(hidden_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.W_s)
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.combined_dim, self.combined_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.combined_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_dim, 1)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
        self.user_transform = nn.Linear(embedding_dim, combined_dim)  
        nn.init.xavier_uniform_(self.user_transform.weight)
        self.user_bn = nn.BatchNorm1d(combined_dim)  

        # Initialize GRU weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)

    def forward(self, combined_his_embeds, user_embed):
        o, _ = self.rnn(combined_his_embeds)  # (batch_size, seq_len, hidden_dim)
        h = torch.matmul(o, self.W_s)  # (batch_size, seq_len, combined_dim)
        user_embed_transformed = self.user_transform(user_embed)  # (batch_size, combined_dim)  
        user_embed_transformed = self.user_bn(user_embed_transformed)
        user_embed_expanded = user_embed_transformed.unsqueeze(1)  # (batch_size, 1, combined_dim)

        diff = h - user_embed_expanded  # (batch_size, seq_len, combined_dim)
        prod = h * user_embed_expanded  # (batch_size, seq_len, combined_dim)
        last_hidden_nn_layer = torch.cat((h, user_embed_expanded.expand_as(h), diff, prod), dim=-1)  # (batch_size, seq_len, 4 * combined_dim)

        last_hidden_nn_layer_reshaped = last_hidden_nn_layer.view(-1, last_hidden_nn_layer.size(-1))
        alpha = self.mlp(last_hidden_nn_layer_reshaped).squeeze(1)  # (batch_size * seq_len)
        alpha = alpha.view(combined_his_embeds.size(0), combined_his_embeds.size(1))  # (batch_size, seq_len)

        a = torch.softmax(alpha, dim=1)  # (batch_size, seq_len)
        z_s = torch.sum(a.unsqueeze(2) * combined_his_embeds, dim=1)  # (batch_size, combined_dim)
        return z_s


def long_term_interest_proxy(combined_his_embeds):
    """
    Calculate the long-term interest proxy using combined embeddings.
    """
    p_l_t = torch.mean(combined_his_embeds, dim=1)
    return p_l_t

def short_term_interest_proxy(combined_his_embeds, discount_factor):
    """
    Calculate the short-term interest proxy using combined embeddings across the entire sequence,
    applying a discount factor to prioritize more recent interactions.
    """
    device = combined_his_embeds.device
    seq_len = combined_his_embeds.size(1)
    time_steps = torch.arange(seq_len, 0, -1, device=device).unsqueeze(0).expand_as(combined_his_embeds[:,:,0])
    discounts = torch.pow(discount_factor, time_steps - 1)  

    discounted_history = combined_his_embeds * discounts.unsqueeze(-1).type_as(combined_his_embeds)  
    p_s_t = discounted_history.sum(dim=1) / discounts.sum(dim=1).unsqueeze(-1)  

    return p_s_t

def bpr_loss(a, positive, negative):
    """
    Simplified BPR loss without the logarithm, for contrastive tasks.

    Parameters:
    - a: The embedding vector (z_l or z_s).
    - positive: The positive proxy or representation (p_l or z_l for long-term, and similarly for short-term).
    - negative: The negative proxy or representation (p_m or z_m for long-term, and similarly for short-term).
    """
    pos_score = torch.sum(a * positive, dim=1)
    neg_score = torch.sum(a * negative, dim=1)
    return F.softplus(neg_score - pos_score)

def calculate_contrastive_loss(z_l, z_s, p_l, p_s):
    """
    Calculate the overall contrastive loss L_con for a user at time t
    """
    # Loss for the long-term and short-term interests pair
    L_con = bpr_loss(z_l, p_l, p_s) + bpr_loss(p_l, z_l, z_s) + \
           bpr_loss(z_s, p_s, p_l) + bpr_loss(p_s, z_s, z_l)

    return L_con

def compute_discrepancy_loss(a, b, discrepancy_weight):
    """
    Calculate the discrepancy loss between two embeddings a and b.
    """
    discrepancy_loss = F.mse_loss(a, b)
    discrepancy_loss = discrepancy_weight * discrepancy_loss
    return discrepancy_loss

class InterestFusionModule(nn.Module):
    def __init__(self, combined_dim, hidden_dim, output_dim, dropout_rate):
        super(InterestFusionModule, self).__init__()
        self.combined_dim = combined_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(self.combined_dim, hidden_dim, batch_first=True)
        
        input_dim_for_alpha = hidden_dim + 2 * combined_dim  

        self.mlp_alpha = nn.Sequential(
            nn.Linear(input_dim_for_alpha, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.mlp_pred = nn.Sequential(
            nn.Linear(2 * combined_dim, output_dim),  
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, combined_his_embeds, z_l, z_s, item_embeds, cat_embeds):
        # Long-term history feature extraction
        h_l, _ = self.gru(combined_his_embeds)
        h_l = h_l[:, -1, :]

        # Attention weights
        alpha = self.mlp_alpha(torch.cat((h_l, z_l, z_s), dim=1))
        z_t = alpha * z_l + (1 - alpha) * z_s

        # Concatenate the embeddings for prediction
        combined_embeddings = torch.cat((z_t, item_embeds, cat_embeds), dim=1)
        y_pred = self.mlp_pred(combined_embeddings)
        return y_pred

class BCELossModule(nn.Module):
    def __init__(self, pos_weight):
        super(BCELossModule, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class PD(nn.Module):
    def __init__(self, num_users, num_items, num_cats, config):
        super(PD, self).__init__()
        self.config = config
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, config.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, config.embedding_dim, padding_idx=0)
        self.combined_dim = 2 * config.embedding_dim

        self.long_term_module = LongTermInterestModule(self.combined_dim, config.embedding_dim, config.dropout_rate)        
        self.short_term_module = ShortTermInterestModule(self.combined_dim, config.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.interest_fusion_module = InterestFusionModule(self.combined_dim, config.hidden_dim, config.output_dim, config.dropout_rate)
        self.bce_loss_module = BCELossModule(pos_weight=torch.tensor([4.0]))
        self.reg_weight = config.reg_weight
        self.discrepancy_weight = config.discrepancy_weight

        with open(config.pd_pop_path, 'rb') as f:
            self.pd_pop_dict = pickle.load(f)
        self.PD_gamma = config.PD_gamma

    def forward(self, stage, batch, device):
        user_ids = batch['user']
        item_ids = batch['item']
        cat_ids = batch['cat']
        unit_time = batch['unit_time']
        items_history_padded = batch['item_his']
        cats_history_padded = batch['cat_his']
        labels = batch['label'].float()        

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        cat_embeds = self.cat_embedding(cat_ids)

        item_his_embeds = self.item_embedding(items_history_padded)
        cat_his_embeds = self.cat_embedding(cats_history_padded)        

        combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)

        z_l = self.long_term_module(combined_his_embeds, user_embeds)
        z_s = self.short_term_module(combined_his_embeds, user_embeds)

        p_l = long_term_interest_proxy(combined_his_embeds)
        p_s = short_term_interest_proxy(combined_his_embeds, self.config.gamma)
        loss_con = calculate_contrastive_loss(z_l, z_s, p_l, p_s)

        y_pred = self.interest_fusion_module(combined_his_embeds, z_l, z_s, item_embeds, cat_embeds)
        labels = labels.view(-1, 1)

        if stage in ['train', 'eval']:
            batch_size = user_ids.size(0)
            pd_popularity = torch.tensor([self.pd_pop_dict[(item_ids[i].item(), unit_time[i].item())] for i in range(batch_size)], device=device).view(-1, 1)
            y_pred = (F.elu(y_pred) + 1) * (pd_popularity ** self.PD_gamma)

        loss_bce = self.bce_loss_module(y_pred, labels)
        loss_discrepancy = compute_discrepancy_loss(z_l, z_s, self.discrepancy_weight)
        regularization_loss = self.reg_weight * sum(torch.norm(param) for param in self.parameters())
        loss = loss_con + loss_bce + loss_discrepancy + regularization_loss
        return loss, y_pred

