import torch
import torch.nn as nn
from config import Config
import torch.nn.functional as F

torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

class ModulePopHistory(nn.Module):
    def __init__(self, config: Config):
        super(ModulePopHistory, self).__init__()
        self.config = config
        self.alpha = self.config.alpha
        self.ema_cache = {}
        self.sigmoid = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

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
        self.relu = nn.ReLU() 

    def forward(self, item_embeds, time_release_embeds, time_embeds):
        temporal_gap = time_release_embeds - time_embeds
        item_temp_embed = torch.cat((temporal_gap, item_embeds, time_embeds, time_release_embeds), 1)
        time_final = self.relu(self.fc_time_value(item_temp_embed))
        return time_final

class ModuleQuality(nn.Module):  
    def __init__(self, config: Config, num_items: int):
        super(ModuleQuality, self).__init__()
        self.config = config
        self.item_quality_embedding = nn.Embedding(num_items + 1, 1, padding_idx=0)

    def forward(self, item_ids):
        quality = self.item_quality_embedding(item_ids)
        return quality

class PopPredict(nn.Module):
    def __init__(self, config: Config, num_items, num_cats, num_stores, max_time):
        super(PopPredict, self).__init__()

        self.config = config
        self.embedding_dim = self.config.embedding_dim

        # Embedding layers
        self.item_embedding = nn.Embedding(num_items + 1, self.embedding_dim, padding_idx=0)
        self.cat_embedding = nn.Embedding(num_cats + 1, self.embedding_dim, padding_idx=0)
        self.store_embedding = nn.Embedding(num_stores + 1, self.embedding_dim)
        self.time_embedding = nn.Embedding(max_time + 1, self.embedding_dim)

        # Modules
        self.module_pop_history = ModulePopHistory(config=config)
        self.module_time = ModuleTime(config=config)
        self.module_quality = ModuleQuality(config=config, num_items=num_items)

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

        # Module outputs
        pop_history_output = self.module_pop_history(pop_histories, item_ids, times)
        time_output = self.module_time(item_embeds, release_time_embeds, time_embeds)
        quality_output = self.module_quality(item_ids)

        normalized_weights = F.softmax(self.attention_weights, dim=0)

        weighted_pop_history_output = pop_history_output * normalized_weights[0]
        weighted_time_output = time_output * normalized_weights[1]
        weighted_quality_output = quality_output * normalized_weights[2]
        output = weighted_pop_history_output + weighted_time_output + weighted_quality_output

        return weighted_pop_history_output, weighted_time_output, weighted_quality_output, output