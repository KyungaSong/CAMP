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

    def ema(self, pop_history_tensor):
        print("pop_history_tensor:\n", pop_history_tensor[0])

        # Calculate weights
        weights = torch.cat((torch.ones(1, device=pop_history_tensor.device), torch.pow(1 - self.alpha, torch.arange(1, pop_history_tensor.shape[1], device=pop_history_tensor.device))))
        print("weights:\n", weights[0])

        # Apply weights
        weighted_pops = torch.zeros_like(pop_history_tensor)
        weighted_pops[:, 0] = pop_history_tensor[:, 0]  
        if pop_history_tensor.shape[1] > 1:  
            weighted_pops[:, 1:] = pop_history_tensor[:, 1:] * (self.config.alpha * weights[1:])
        print("weighted_pops:\n", weighted_pops[0])

        # Calculate cumulative sums and normalization
        cumulative_pops = torch.cumsum(weighted_pops, dim=1)
        normalization = torch.cumsum(weights, dim=0)

        # Normalize the EMA
        ema_pops = cumulative_pops / normalization
        print("ema_pops:\n", ema_pops[0])
        return ema_pops


    def forward(self, pop_history_tensor):
        history_ema = self.ema(pop_history_tensor)
        return history_ema


class ModuleTime(nn.Module):
    def __init__(self, config: Config):
        super(ModuleTime, self).__init__()
        self.config = config
        self.fc_item_pop_value = nn.Linear(config.embedding_dim*4, 1)
        self.relu = nn.ReLU()

    def forward(self, item_embeds, time_release_embeds, time_embeds):
        temporal_dis = time_release_embeds - time_embeds
        item_temp_joint_embed = torch.cat((temporal_dis, item_embeds, time_embeds, time_release_embeds), 1)
        joint_item_temp_value = self.relu(self.fc_item_pop_value(item_temp_joint_embed))
        return joint_item_temp_value

class ModuleSideInfo(nn.Module):
    def __init__(self, config: Config):
        super(ModuleSideInfo, self).__init__()
        self.config = config
        if self.config.is_douban:
            self.fc_output = nn.Linear(3*config.embedding_dim, 1)
        else:
            self.fc_output = nn.Linear(2*config.embedding_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, genre_embeds, director_embeds=None, actor_embeds=None):
        genre_embeds = genre_embeds.mean(dim=1)
        if director_embeds is not None and actor_embeds is not None:
            actor_embeds = actor_embeds.mean(dim=1)
            embed_sideinfo = torch.cat((genre_embeds, director_embeds, actor_embeds), 1)
        else:
            embed_sideinfo = torch.cat((genre_embeds, director_embeds), 1)
        output = self.relu(self.fc_output(embed_sideinfo))
        return output


class PopPredict(nn.Module):
    def __init__(self, is_training, config: Config, num_item, max_time):
        super(PopPredict, self).__init__()

        self.config = config
        self.is_training = is_training
        # num_genre = self.config.num_side_info[0]
        # num_director = self.config.num_side_info[1] if self.config.is_douban else 1
        # num_actor = self.config.num_side_info[2] if self.config.is_douban else 1

        self.embedding_dim = config.embedding_dim

        # Embedding layers
        self.item_embedding = nn.Embedding(num_item, self.embedding_dim)
        self.time_embedding = nn.Embedding(max_time + 1, self.embedding_dim)
        # self.genre_embedding = nn.Embedding(num_genre, self.embedding_dim, padding_idx=0)
        # self.director_embedding = nn.Embedding(num_director, self.embedding_dim, padding_idx=0)
        # self.actor_embedding = nn.Embedding(num_actor, self.embedding_dim, padding_idx=0)

        # Modules
        self.module_pop_history = ModulePopHistory(config=config)
        # self.module_sideinfo = ModuleSideInfo(config=config)
        # self.module_time = ModuleTime(config=config)

        # Attention mechanism
        self.attention_weights = nn.Parameter(torch.ones(3, 1) / 3)  # Adjusted for 3 modules
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # def forward(self, item, time_release, item_genre, item_director, item_actor, time, pop_history, pop_gt, valid_pop_len):
    #     item_embeds = self.item_embedding(item)
    def forward(self, pop_history):
        # item_embeds = self.item_embedding(item)
        # time_release_embeds = self.time_embedding(release_time)
        # genre_embeds = self.genre_embedding(item_genre)
        # time_embeds = self.time_embedding(time)
        
        # director_embeds = self.director_embedding(item_director) if self.config.is_douban else torch.zeros_like(genre_embeds)
        # actor_embeds = self.actor_embedding(item_actor) if self.config.is_douban else torch.zeros_like(genre_embeds)

        # Module outputs
        pop_history_output = self.module_pop_history(pop_history)
        # time_output = self.module_time(item_embeds, time_release_embeds, time_embeds)
        
        # if self.config.is_douban:
        #     sideinfo_output = self.module_sideinfo(genre_embeds, director_embeds, actor_embeds)
        # else:
        #     sideinfo_output = self.module_sideinfo(genre_embeds)

        # # Concatenate module outputs without the periodic module
        # pred_all = torch.cat((pop_history_output, time_output, sideinfo_output), 1)

        # # Apply attention weights (adjusted for 3 modules)
        # normalized_weights = F.softmax(self.attention_weights, dim=0)
        # output = torch.mm(pred_all, normalized_weights).squeeze()

        # if not self.is_training:
        #     print('Attention weights:', normalized_weights.data.cpu().numpy())

        # return pop_history_output, time_output, sideinfo_output, output
        print("pop_history_output:\n", pop_history_output[0])
        return pop_history_output
