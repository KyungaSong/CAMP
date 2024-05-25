import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

class ModulePopHistory(nn.Module):
    def __init__(self, config):
        super(ModulePopHistory, self).__init__()
        self.config = config
        self.alpha = self.config.alpha
        self.ema_cache = {}
        self.sigmoid = nn.Sigmoid()

    def ema(self, pop_history, item_id):
        if item_id in self.ema_cache:
            return self.ema_cache[item_id]

        alpha = self.alpha
        ema_all = torch.zeros_like(pop_history, dtype=torch.float)
        ema_all[:, 0] = pop_history[:, 0]

        for t in range(1, pop_history.size(1)):
            ema_all[:, t] = (1 - alpha) * ema_all[:, t-1] + alpha * pop_history[:, t]

        self.ema_cache[item_id] = ema_all
        return ema_all

    def forward(self, pop_history, item_id, time):
        history_ema = self.ema(pop_history, item_id)
        time_before = time - 1
        time_before_clamped = torch.clamp(time_before, min=0)
        history_final = torch.gather(history_ema, 1, time_before_clamped.long().unsqueeze(1))
        return history_final

class ModuleTime(nn.Module):
    def __init__(self, config: Config):
        super(ModuleTime, self).__init__()
        self.config = config
        self.fc_time_value = nn.Linear(config.embedding_dim * 4, 1)
        self.batch_norm = nn.BatchNorm1d(config.embedding_dim * 4)
        self.relu = nn.ReLU() 
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, item_embeds, time_release_embeds, time_embeds):
        temporal_gap = time_release_embeds - time_embeds
        item_temp_embed = torch.cat((temporal_gap, item_embeds, time_embeds, time_release_embeds), 1)
        item_temp_embed = self.batch_norm(item_temp_embed)  
        item_temp_embed = self.dropout(item_temp_embed)
        time_final = self.relu(self.fc_time_value(item_temp_embed))
        return time_final

class ModuleSideInfo(nn.Module):
    def __init__(self, config: Config):
        super(ModuleSideInfo, self).__init__()
        self.config = config
        self.fc_output = nn.Linear(2 * config.embedding_dim, 1)
        self.batch_norm = nn.BatchNorm1d(2 * config.embedding_dim)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, cat_embeds, store_embeds):
        embed_sideinfo = torch.cat((cat_embeds, store_embeds), 1)
        embed_sideinfo = self.batch_norm(embed_sideinfo)  
        embed_sideinfo = self.dropout(embed_sideinfo)
        embed_sideinfo = self.fc_output(embed_sideinfo)
        return embed_sideinfo

class PopPredict(nn.Module):
    def __init__(self, is_training, config: Config, num_items, num_cats, num_stores, max_time):
        super(PopPredict, self).__init__()

        self.config = config
        self.is_training = is_training
        self.embedding_dim = self.config.embedding_dim

        # Embedding layers
        self.item_embedding = nn.Embedding(num_items + 1, self.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats + 1, self.embedding_dim, padding_idx=0)
        self.store_embedding = nn.Embedding(num_stores, self.embedding_dim)
        self.time_embedding = nn.Embedding(max_time + 1, self.embedding_dim)

        # Modules
        self.module_pop_history = ModulePopHistory(config=config)
        self.module_time = ModuleTime(config=config)
        self.module_sideinfo = ModuleSideInfo(config=config)

        # Attention mechanism
        self.attention_weights = nn.Parameter(torch.ones(3, 1) / 3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch):
        item_ids = batch['item']
        times = batch['time']
        release_times = batch['release_time']
        pop_histories = batch['pop_history']
        categories = batch['category']
        stores = batch['store']

        item_embeds = self.item_embedding(item_ids)
        time_embeds = self.time_embedding(times)
        release_time_embeds = self.time_embedding(release_times)
        cat_embeds = self.cat_embedding(categories)
        store_embeds = self.store_embedding(stores)

        # Module outputs
        pop_history_output = self.module_pop_history(pop_histories, item_ids, times)
        time_output = self.module_time(item_embeds, release_time_embeds, time_embeds)
        sideinfo_output = self.module_sideinfo(cat_embeds, store_embeds)

        normalized_weights = F.softmax(self.attention_weights, dim=0)

        weighted_pop_history_output = pop_history_output * normalized_weights[0]
        weighted_time_output = time_output * normalized_weights[1]
        weighted_sideinfo_output = sideinfo_output * normalized_weights[2]
        output = weighted_pop_history_output + weighted_time_output + weighted_sideinfo_output

        # if not self.is_training:
        #     print('Attention weights:', normalized_weights.data.cpu().numpy())
        return weighted_pop_history_output, weighted_time_output, weighted_sideinfo_output, output
