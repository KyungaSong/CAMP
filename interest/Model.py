import pickle
import bisect

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def contrastive_loss(z_l, z_s, p_l, p_s):
    """
    Calculate the overall contrastive loss L_con for a user at time t
    """
    # Loss for the long-term and short-term interests pair
    L_con = bpr_loss(z_l, p_l, p_s) + bpr_loss(p_l, z_l, z_s) + \
           bpr_loss(z_s, p_s, p_l) + bpr_loss(p_s, z_s, z_l)

    return L_con

def discrepancy_loss(a, b, discrepancy_weight):
    """
    Calculate the discrepancy loss between two embeddings a and b.
    """
    discrepancy_loss = F.mse_loss(a, b)
    discrepancy_loss = discrepancy_weight * discrepancy_loss
    return discrepancy_loss

class InterestFusionModule(nn.Module):
    def __init__(self, combined_dim, hidden_dim, output_dim, dropout_rate, wo_con, wo_qlt):
        super(InterestFusionModule, self).__init__()
        self.combined_dim = combined_dim
        self.hidden_dim = hidden_dim
        self.wo_con = wo_con
        self.wo_qlt = wo_qlt
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

    def forward(self, combined_his_embeds, z_l, z_s, item_embeds, cat_embeds, con_embeds, qlt_embeds):
        # Long-term history feature extraction
        h_l, _ = self.gru(combined_his_embeds)
        h_l = h_l[:, -1, :]

        # Attention weights
        alpha = self.mlp_alpha(torch.cat((h_l, z_l, z_s), dim=1))
        z_t = alpha * z_l + (1 - alpha) * z_s

        # Concatenate the embeddings for prediction
        if self.wo_con and self.wo_qlt:
            combined_embeddings = torch.cat((z_t, item_embeds, cat_embeds), dim=1)
        elif self.wo_con:
            combined_embeddings = torch.cat((z_t, item_embeds, cat_embeds, qlt_embeds), dim=1)
        elif self.wo_qlt:
            combined_embeddings = torch.cat((z_t, item_embeds, cat_embeds, con_embeds), dim=1)
        else:            
            combined_embeddings = torch.cat((z_t, item_embeds, cat_embeds, con_embeds, qlt_embeds), dim=1)
        y_pred = self.mlp_pred(combined_embeddings)
        return y_pred

class BCELossModule(nn.Module):
    def __init__(self, pos_weight):
        super(BCELossModule, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

class CAMP(nn.Module):
    def __init__(self, num_users, num_items, num_cats, config):
        super(CAMP, self).__init__()
        self.config = config
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, config.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, config.embedding_dim, padding_idx=0)
        self.con_transform = nn.Linear(1, config.embedding_dim)
        self.qlt_transform = nn.Linear(1, config.embedding_dim)
        if config.wo_con and config.wo_qlt:
            self.combined_dim = 2 * config.embedding_dim
        elif config.wo_con or config.wo_qlt:
            self.combined_dim = 3 * config.embedding_dim
        else:
            self.combined_dim = 4 * config.embedding_dim
        self.long_term_module = LongTermInterestModule(self.combined_dim, config.embedding_dim, config.dropout_rate)        
        self.short_term_module = ShortTermInterestModule(self.combined_dim, config.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.interest_fusion_module = InterestFusionModule(self.combined_dim, config.hidden_dim, config.output_dim, config.dropout_rate, config.wo_con, config.wo_qlt)
        self.bce_loss_module = BCELossModule(pos_weight=torch.tensor([4.0]))
        self.reg_weight = config.reg_weight
        self.discrepancy_weight = config.discrepancy_weight

        self.is_testing = False
    
    def set_testing_mode(self, is_testing):
        self.is_testing = is_testing

    def forward(self, batch, device):
        user_ids = batch['user']
        item_ids = batch['item']
        cat_ids = batch['cat']
        con = batch['con']
        qlt = batch['qlt']
        items_history_padded = batch['item_his']
        cats_history_padded = batch['cat_his']
        con_his = batch['con_his']
        qlt_his = batch['qlt_his']
        labels = batch['label'].float()

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        cat_embeds = self.cat_embedding(cat_ids)
        con_embeds = self.con_transform(con.unsqueeze(-1))
        qlt_embeds = self.qlt_transform(qlt.unsqueeze(-1))

        item_his_embeds = self.item_embedding(items_history_padded)
        cat_his_embeds = self.cat_embedding(cats_history_padded)
        con_his_embeds = self.con_transform(con_his.unsqueeze(-1))
        qlt_his_embeds = self.qlt_transform(qlt_his.unsqueeze(-1))         

        if self.config.wo_con and self.config.wo_qlt:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        elif self.config.wo_con:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, qlt_his_embeds), dim=-1)
        elif self.config.wo_qlt:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, con_his_embeds), dim=-1)
        else:            
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, con_his_embeds, qlt_his_embeds), dim=-1)

        z_l = self.long_term_module(combined_his_embeds, user_embeds)
        z_s = self.short_term_module(combined_his_embeds, user_embeds)

        p_l = long_term_interest_proxy(combined_his_embeds)
        p_s = short_term_interest_proxy(combined_his_embeds, self.config.gamma)
        loss_con = contrastive_loss(z_l, z_s, p_l, p_s)

        y_pred = self.interest_fusion_module(combined_his_embeds, z_l, z_s, item_embeds, cat_embeds, con_embeds, qlt_embeds)
        labels = labels.view(-1, 1)
        loss_bce = self.bce_loss_module(y_pred, labels)
        loss_discrepancy = discrepancy_loss(z_l, z_s, self.discrepancy_weight)
        regularization_loss = self.reg_weight * sum(torch.norm(param) for param in self.parameters())
        loss = loss_con + loss_bce + loss_discrepancy + regularization_loss
        return loss, y_pred
    

class CAMP_T(nn.Module):
    def __init__(self, num_users, num_items, num_cats, config):
        super(CAMP_T, self).__init__()
        self.config = config
        self.user_embedding = nn.Embedding(num_users, config.embedding_dim)
        self.item_embedding = nn.Embedding(num_items, config.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats, config.embedding_dim, padding_idx=0)
        self.con_transform = nn.Linear(1, config.embedding_dim)
        self.qlt_transform = nn.Linear(1, config.embedding_dim)

        if config.wo_con and config.wo_qlt:
            self.combined_dim = 2 * config.embedding_dim
        elif config.wo_con or config.wo_qlt:
            self.combined_dim = 3 * config.embedding_dim
        else:
            self.combined_dim = 4 * config.embedding_dim

        self.long_term_module = LongTermInterestModule(self.combined_dim, config.embedding_dim, config.dropout_rate)        
        self.short_term_module = ShortTermInterestModule(self.combined_dim, config.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.interest_fusion_module = InterestFusionModule(self.combined_dim, config.hidden_dim, config.output_dim, config.dropout_rate, config.wo_con, config.wo_qlt)
        self.bce_loss_module = BCELossModule(pos_weight=torch.tensor([4.0]))
        self.reg_weight = config.reg_weight
        self.discrepancy_weight = config.discrepancy_weight

        self.beta = nn.Parameter(torch.ones(num_items, ) * config.TIDE_beta)
        self.tau = torch.ones(num_items, ) * config.TIDE_tau
        self.item_quality = nn.Parameter(torch.ones(num_items, ) * config.TIDE_q)

        with open(config.tide_con_path, 'rb') as f:
            self.conformity_dict = pickle.load(f)

        self.item_timestamps = {item_id: sorted([t for (_item, t) in self.conformity_dict.keys() if item_id == _item]) for item_id in range(num_items)}

        self.is_testing = False
    
    def set_testing_mode(self, is_testing):
        self.is_testing = is_testing
    
    def get_nearest_conformity(self, item_id, timestamp):
        timestamps = self.item_timestamps[item_id]
        pos = bisect.bisect_left(timestamps, timestamp)

        if pos == 0:
            nearest_timestamp = timestamps[0]
        elif pos == len(timestamps):
            nearest_timestamp = timestamps[-1]
        else:
            before = timestamps[pos - 1]
            after = timestamps[pos]
            nearest_timestamp = before if abs(timestamp - before) < abs(timestamp - after) else after

        return self.conformity_dict[(item_id, nearest_timestamp)]

    def forward(self, batch, device):
        user_ids = batch['user']
        item_ids = batch['item']
        cat_ids = batch['cat']
        items_history_padded = batch['item_his']
        cats_history_padded = batch['cat_his']
        labels = batch['label'].float()
        timestamps = batch['timestamp']

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        cat_embeds = self.cat_embedding(cat_ids)

        item_his_embeds = self.item_embedding(items_history_padded)
        cat_his_embeds = self.cat_embedding(cats_history_padded)        

        batch_size, seq_len = items_history_padded.size()

        # Calculate conformity and quality for each item in the history
        conformity_his = torch.zeros(batch_size, seq_len, 1, device=device)
        quality_his = torch.zeros(batch_size, seq_len, 1, device=device)

        for i in range(batch_size):
            for j in range(seq_len):
                if items_history_padded[i, j] != 0: 
                    item_id = items_history_padded[i, j].item()
                    timestamp = timestamps[i].item()
                    conformity_his[i, j] = F.softplus(self.beta[item_id]) * torch.tensor(
                        self.get_nearest_conformity(item_id, timestamp), 
                        device=device, 
                        dtype=torch.float
                    )
                    quality_his[i, j] = F.softplus(self.item_quality[item_id])

        # Transform conformity and quality to embedding dimensions
        conformity_his_embeds = self.con_transform(conformity_his)  # (batch_size, seq_len, embedding_dim)
        quality_his_embeds = self.qlt_transform(quality_his)  # (batch_size, seq_len, embedding_dim)

        # Combine embeddings based on the configuration
        if self.config.wo_con and self.config.wo_qlt:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds), dim=-1)
        elif self.config.wo_con:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, quality_his_embeds), dim=-1)
        elif self.config.wo_qlt:
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, conformity_his_embeds), dim=-1)
        else:            
            combined_his_embeds = torch.cat((item_his_embeds, cat_his_embeds, conformity_his_embeds, quality_his_embeds), dim=-1)

        z_l = self.long_term_module(combined_his_embeds, user_embeds)
        z_s = self.short_term_module(combined_his_embeds, user_embeds)

        p_l = long_term_interest_proxy(combined_his_embeds)
        p_s = short_term_interest_proxy(combined_his_embeds, self.config.gamma)
        loss_con = contrastive_loss(z_l, z_s, p_l, p_s)

        # Calculate conformity and quality for the current item
        quality = F.softplus(self.item_quality[item_ids])
        conformity = F.softplus(self.beta[item_ids]) * torch.tensor(
            [self.get_nearest_conformity(item_ids[i].item(), timestamps[i].item()) 
            for i in range(batch_size)], 
            device=device,
            dtype=torch.float
        )
        # if self.is_testing:
        #     conformity = conformity * 0.4

        # Transform current conformity and quality to embedding dimensions
        con_embeds = self.con_transform(conformity.unsqueeze(-1))
        qlt_embeds = self.qlt_transform(quality.unsqueeze(-1))

        y_pred = self.interest_fusion_module(combined_his_embeds, z_l, z_s, item_embeds, cat_embeds, con_embeds, qlt_embeds)
        labels = labels.view(-1, 1)
        loss_bce = self.bce_loss_module(y_pred, labels)

        loss_discrepancy = discrepancy_loss(z_l, z_s, self.discrepancy_weight)

        regularization_loss = self.reg_weight * sum(torch.norm(param) for param in self.parameters())

        loss = loss_con + loss_bce + loss_discrepancy + regularization_loss
        return loss, y_pred