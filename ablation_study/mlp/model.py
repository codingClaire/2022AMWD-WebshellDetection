import torch
from torch_geometric.nn import (
    global_mean_pool,global_add_pool,global_max_pool
)


class EdgeEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        self.feature_layer = torch.nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.feature_layer(x)


class MLPModel(torch.nn.Module):
    def __init__(self, params):
        super(MLPModel, self).__init__()
        self.emb_dim = params["emb_dim"]
        self.pool = params["pooling"]
        self.type_nums = params["type_nums"]
        self.value_nums =params["value_nums"]
        self.type_encoder = torch.nn.Embedding(self.type_nums, self.emb_dim)
        self.value_encoder = torch.nn.Embedding(self.value_nums, self.emb_dim)
        self.predict = torch.nn.Linear(2*self.emb_dim, 1)
    
    
    def forward(self, type, value, batch):
        ### feature embedding 
        h = torch.cat((self.type_encoder(type),self.value_encoder(value)),dim=1)
        if self.pool == "mean":
            graph_emb = global_mean_pool(h, batch)
        elif self.pool == "max":
            graph_emb = global_max_pool(h,batch)
        elif self.pool == "sum":
            graph_emb = global_add_pool(h,batch)
        return self.predict(graph_emb)