import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import (
    global_mean_pool,global_max_pool,global_add_pool
)


class EdgeEncoder(torch.nn.Module):
    def __init__(self, in_dim, emb_dim):
        super(EdgeEncoder, self).__init__()
        self.feature_layer = torch.nn.Linear(in_dim, emb_dim)

    def forward(self, x):
        return self.feature_layer(x)


class GCNConv(MessagePassing):
    def __init__(self, edge_dim, emb_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        self.edge_encoder = EdgeEncoder(edge_dim, emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        if edge_attr != None:
            # for have edge_attr situation
            edge_embedding = self.edge_encoder(edge_attr)
            return self.propagate(
                edge_index, x=x, edge_attr=edge_embedding, norm=norm
            ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)
        else:
            # for no edge_attr situation
            edge_embedding = 0
            return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_embedding, use_edge_attr=False) + F.relu(
                x + self.root_emb.weight
            ) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        if edge_attr != None:
            # for have edge_attr situation
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        else:
            # for no edge_attr situation
            return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out

class GINConv(MessagePassing):
    def __init__(self, edge_dim,emb_dim):
        super(GINConv, self).__init__(aggr = "add")

        self.mlp = torch.nn.Sequential( torch.nn.Linear(emb_dim, 2*emb_dim), 
                                        torch.nn.BatchNorm1d(2*emb_dim), 
                                        torch.nn.ReLU(), 
                                        torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.edge_encoder = EdgeEncoder(edge_dim,emb_dim)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr != None:
            # for have edge_attr situation  
            edge_embedding = self.edge_encoder(edge_attr)
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, use_edge_attr = True))
        else:
            # for no edge_attr situation 
            edge_embedding = 0 # not really use
            out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding, use_edge_attr = False))
        return out

    def message(self, x_j, edge_attr,use_edge_attr):
        if use_edge_attr != False:
            # for have edge_attr situation  
            return F.relu(x_j + edge_attr)
        else:
            # for no edge_attr situation  
            return F.relu(x_j)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GnnModel(torch.nn.Module):
    def __init__(self, params):
        super(GnnModel, self).__init__()
        self.num_layer =  params["num_layer"]
        self.drop_ratio = params["drop_ratio"]
        self.JK = params["JK"]
        self.residual = params["residual"]
        self.gnn_type = params["gnn_type"]
        self.emb_dim =  2*params["emb_dim"]
        if "edge_dim" in params.keys():
            # currently only for ogbg-code2 dataset
            self.edge_dim  = params["edge_dim"]
        else:
            # for this situation in conv layer
            # edge_dim is useless
            self.edge_dim  = params["emb_dim"]
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for _ in range(self.num_layer):
            if self.gnn_type == 'gin':
                self.convs.append(GINConv(self.edge_dim,self.emb_dim))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(self.edge_dim,self.emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(self.gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(self.emb_dim))

    def forward(self, input_feature, edge_index,edge_attr):
        ### computing input node embedding
        h_list = [input_feature]
        for layer in range(self.num_layer):
            # conv --> batchnorm --> relu --> dropout
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h =h+ h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation = node_representation+h_list[layer]

        return node_representation


class predictModel(torch.nn.Module):
    def __init__(self, params):
        super(predictModel, self).__init__()
        self.last_pool=params["last_pooling"]
        self.emb_dim = params["emb_dim"]
        self.type_nums = params["type_nums"]
        self.value_nums =params["value_nums"]
        self.type_encoder = torch.nn.Embedding(self.type_nums, self.emb_dim)
        self.value_encoder = torch.nn.Embedding(self.value_nums, self.emb_dim)
        self.gnn = GnnModel(params)
        self.predict = torch.nn.Linear(2*self.emb_dim, 1)
    
    
    def forward(self, type, value, edge_index,edge_attr,batch):
        ### feature embedding 
        h = torch.cat((self.type_encoder(type),self.value_encoder(value)),dim=1)
        node_emb = self.gnn(h,edge_index, edge_attr)
        if self.last_pool=='mean':
            graph_emb = global_mean_pool(node_emb, batch)
        elif self.last_pool=='max':
            graph_emb = global_max_pool(node_emb, batch)
        elif self.last_pool=='add':
            graph_emb = global_add_pool(node_emb, batch)

        return self.predict(graph_emb)