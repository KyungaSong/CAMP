import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

def base_bpr_loss(a, positive, negative):
    """
    Simplified BPR loss without the logarithm, for contrastive tasks.

    Parameters:
    - a: The embedding vector (z_l, z_s, p_l or p_s).
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
    L_con = base_bpr_loss(z_l, p_l, p_s) + base_bpr_loss(p_l, z_l, z_s) + \
           base_bpr_loss(z_s, p_s, p_l) + base_bpr_loss(p_s, z_s, z_l)

    return L_con

def discrepancy_loss(a, b, discrepancy_weight):
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

class DICE(nn.Module):
    def __init__(self, num_users, num_items, num_cats, config):
        super(DICE, self).__init__()
        self.gamma = config.gamma
        self.embedding_dim = config.embedding_dim
        self.combined_dim = 2 * self.embedding_dim
        self.users_int = nn.Embedding(num_users, self.embedding_dim)
        self.users_pop = nn.Embedding(num_users, self.embedding_dim)
        self.items_int = nn.Embedding(num_items, self.embedding_dim, padding_idx=0)
        self.items_pop = nn.Embedding(num_items, self.embedding_dim, padding_idx=0)
        self.cats_int = nn.Embedding(num_cats, self.embedding_dim, padding_idx=0)        
        self.cats_pop = nn.Embedding(num_cats, self.embedding_dim, padding_idx=0) 

        self.long_term_module = LongTermInterestModule(self.combined_dim, self.embedding_dim, config.dropout_rate)        
        self.short_term_module = ShortTermInterestModule(self.combined_dim, self.embedding_dim, config.hidden_dim, config.dropout_rate)
        self.interest_fusion_module = InterestFusionModule(self.combined_dim, config.hidden_dim, config.output_dim, config.dropout_rate)

        self.bce_loss_module = BCELossModule(pos_weight=torch.tensor([4.0]))
        self.int_weight = config.int_weight
        self.pop_weight = config.pop_weight
        self.reg_weight = config.reg_weight
        self.discrepancy_weight = config.discrepancy_weight
        self.discrepancy_type = config.discrepancy_type

        if self.discrepancy_type == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif self.discrepancy_type == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif self.discrepancy_type == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.init_params()

        self.is_testing = False
    
    def set_testing_mode(self, is_testing):
        self.is_testing = is_testing

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    def dcor(self, x, y):

        a = torch.norm(x[:,None] - x, p = 2, dim = 2)
        b = torch.norm(y[:,None] - y, p = 2, dim = 2)

        A = a - a.mean(dim=0)[None,:] - a.mean(dim=1)[:,None] + a.mean()
        B = b - b.mean(dim=0)[None,:] - b.mean(dim=1)[:,None] + b.mean() 

        n = x.size(0)

        dcov2_xy = (A * B).sum()/float(n * n)
        dcov2_xx = (A * A).sum()/float(n * n)
        dcov2_yy = (B * B).sum()/float(n * n)
        dcor = -torch.sqrt(dcov2_xy)/torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.users_int.weight.size(1))
        self.users_int.weight.data.uniform_(-stdv, stdv)
        self.users_pop.weight.data.uniform_(-stdv, stdv)
        self.items_int.weight.data.uniform_(-stdv, stdv)
        self.items_pop.weight.data.uniform_(-stdv, stdv)
        self.cats_int.weight.data.uniform_(-stdv, stdv)
        self.cats_pop.weight.data.uniform_(-stdv, stdv)

    def bpr_loss(self, pos_y_pred, neg_y_pred):

        return -torch.mean(torch.log(torch.sigmoid(pos_y_pred - neg_y_pred)))
    
    def mask_bpr_loss(self, pos_y_pred, neg_y_pred, mask):

        return -torch.mean(mask*torch.log(torch.sigmoid(pos_y_pred - neg_y_pred)))

    def forward(self, batch, device):        
        user_ids = batch['user']
        pos_item = batch['pos_item']        
        pos_cat = batch['pos_cat']
        neg_item = batch['neg_item']
        neg_cat = batch['neg_cat']
        mask = batch['mask']
        items_history_padded = batch['item_his']
        cats_history_padded = batch['cat_his']

        user_int = self.users_int(user_ids)
        user_pop = self.users_pop(user_ids)  
        pos_item_int = self.items_int(pos_item)        
        pos_item_pop = self.items_pop(pos_item)
        neg_item_int = self.items_int(neg_item)
        neg_item_pop = self.items_pop(neg_item)
        pos_cat_int = self.cats_int(pos_cat)
        pos_cat_pop = self.cats_pop(pos_cat)
        neg_cat_int = self.cats_int(neg_cat)
        neg_cat_pop = self.cats_pop(neg_cat)

        combined_his_embeds_int = torch.cat((self.items_int(items_history_padded), self.cats_int(cats_history_padded)), dim=-1)
        combined_his_embeds_pop = torch.cat((self.items_pop(items_history_padded), self.cats_pop(cats_history_padded)), dim=-1)   

        z_l_int = self.long_term_module(combined_his_embeds_int, user_int)
        z_s_int = self.short_term_module(combined_his_embeds_int, user_int)
        z_l_pop = self.long_term_module(combined_his_embeds_pop, user_pop)
        z_s_pop = self.short_term_module(combined_his_embeds_pop, user_pop)

        p_l_int = long_term_interest_proxy(combined_his_embeds_int)
        p_s_int = short_term_interest_proxy(combined_his_embeds_int, self.gamma)
        p_l_pop = long_term_interest_proxy(combined_his_embeds_pop)
        p_s_pop = short_term_interest_proxy(combined_his_embeds_pop, self.gamma)

        loss_con = contrastive_loss(z_l_int, z_s_int, p_l_int, p_s_int) + contrastive_loss(z_l_pop, z_s_pop, p_l_pop, p_s_pop)
        loss_discrepancy_base = discrepancy_loss(z_l_int, z_s_int, self.discrepancy_weight) + discrepancy_loss(z_l_pop, z_s_pop, self.discrepancy_weight)
        reg_loss = self.reg_weight * sum(torch.norm(param) for param in self.parameters())
        
        loss_base = loss_con + loss_discrepancy_base + reg_loss
        
        # Intergrate DICE method
        pos_y_pred_int = self.interest_fusion_module(combined_his_embeds_int, z_l_int, z_s_int, pos_item_int, pos_cat_int)
        neg_y_pred_int = self.interest_fusion_module(combined_his_embeds_int, z_l_int, z_s_int, neg_item_int, neg_cat_int)
        pos_y_pred_pop = self.interest_fusion_module(combined_his_embeds_pop, z_l_pop, z_s_pop, pos_item_pop, pos_cat_pop)
        neg_y_pred_pop = self.interest_fusion_module(combined_his_embeds_pop, z_l_pop, z_s_pop, neg_item_pop, neg_cat_pop)

        pos_y_pred_total = (pos_y_pred_int + pos_y_pred_pop)/2
        neg_y_pred_total = (neg_y_pred_int + neg_y_pred_pop)/2

        loss_int = self.mask_bpr_loss(pos_y_pred_int, neg_y_pred_int, mask)
        loss_pop = self.mask_bpr_loss(neg_y_pred_pop, pos_y_pred_pop, mask) + self.mask_bpr_loss(pos_y_pred_pop, neg_y_pred_pop, ~mask)
        loss_click = self.bpr_loss(pos_y_pred_total, neg_y_pred_total)

        item_all = torch.unique(torch.cat((pos_item, neg_item)))
        item_int = self.items_int(item_all)
        item_pop = self.items_pop(item_all)
        user_all = torch.unique(user_ids)
        user_int = self.users_int(user_all)
        user_pop = self.users_pop(user_all)
        loss_discrepancy = self.criterion_discrepancy(item_int, item_pop) + self.criterion_discrepancy(user_int, user_pop)

        loss = loss_base + self.int_weight * loss_int + self.pop_weight * loss_pop + loss_click - self.discrepancy_weight * loss_discrepancy

        return loss, pos_y_pred_total, neg_y_pred_total


