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
        return self.sigmoid(history_final)

class ModuleTime(nn.Module):
    def __init__(self, config: Config):
        super(ModuleTime, self).__init__()
        self.config = config
        self.fc_item_pop_value = nn.Linear(config.embedding_dim*4, 1)
        self.relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, item_embeds, time_release_embeds, time_embeds):
        temporal_gap = time_release_embeds - time_embeds
        item_temp_joint_embed = torch.cat((temporal_gap, item_embeds, time_embeds, time_release_embeds), 1)
        time_final = self.relu(self.fc_item_pop_value(item_temp_joint_embed))
        return self.sigmoid(time_final)

class ModuleSideInfo(nn.Module):
    def __init__(self, config: Config):
        super(ModuleSideInfo, self).__init__()
        self.config = config
        self.rating_number_fc = nn.Linear(1, config.embedding_dim)
        self.fc_output = nn.Linear(3 * config.embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rating_number, cat_embeds, store_embeds):
        rating_number = self.rating_number_fc(rating_number.view(-1, 1))
        embed_sideinfo = torch.cat((rating_number, cat_embeds, store_embeds), 1)    
        embed_sideinfo = self.fc_output(embed_sideinfo)
        return self.sigmoid(embed_sideinfo)

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
        rating_numbers = batch['rating_number']
        categories = batch['category']
        stores = batch['store']

        item_embeds = self.item_embedding(item_ids)
        time_embeds = self.time_embedding(times)
        release_time_embeds = self.time_embedding(release_times)        
        cat_embeds = self.cat_embedding(categories)
        store_embeds = self.store_embedding(stores)

        # Module outputs
        # pop_history_output = self.module_pop_history(pop_histories)
        pop_history_output = self.module_pop_history(pop_histories, item_ids, times)
        time_output = self.module_time(item_embeds, release_time_embeds, time_embeds)        
        sideinfo_output = self.module_sideinfo(rating_numbers, cat_embeds, store_embeds)
        # print(f'pop: {pop_history_output.size()}, time: {time_output.size()}, side: {sideinfo_output.size()}')

        # Concatenate module outputs without the periodic module
        pred_all = torch.cat((pop_history_output, time_output, sideinfo_output), 1)

        # Apply attention weights (adjusted for 3 modules)
        normalized_weights = F.softmax(self.attention_weights, dim=0)
        output = torch.mm(pred_all, normalized_weights).squeeze()

        if not self.is_training:
            print('Attention weights:', normalized_weights.data.cpu().numpy())
        return pop_history_output, time_output, sideinfo_output, output
        
