
from torch_geometric.data import Dataset as geometricDataset
from torch_geometric.data import Data
import pandas as pd
import os
import torch

def get_load_dataset(data_path, task,train_num,val_num):
    def load_graph_dataset():
        return ASTDataset(data_path,task,train_num,val_num)
    return  load_graph_dataset


class ASTDataset(geometricDataset):
    def __init__(self, data_path,task, train_num,val_num):
        self.data_path = data_path
        self.data_len = len(os.listdir(data_path))
        self.task = task
        self.train_num = train_num
        self.val_num = val_num
    
    def __getitem__(self, index):
        if self.task == "val":
            act_index = self.train_num + index
            dic = torch.load(self.data_path + str(act_index) + ".pt")
            label_df = pd.read_csv("/home/jliao/webshell/tctrain/train.csv")
            y = 0 if label_df["label"][act_index] == "white" else 1
            return Data(num_nodes=dic["node_type"].shape[0], 
                        node_type= dic["node_type"], 
                        node_value = dic["node_value"],
                        edge_index = dic["edge_index"],
                        edge_attr=dic["edge_attr"],
                        y = y)
        elif self.task == "train":
            dic = torch.load(self.data_path + str(index) + ".pt")
            label_df = pd.read_csv("/home/jliao/webshell/tctrain/train.csv")
            y = 0 if label_df["label"][index] == "white" else 1
            return Data(num_nodes=dic["node_type"].shape[0], 
                        node_type= dic["node_type"], 
                        node_value = dic["node_value"],
                        edge_index = dic["edge_index"],
                        edge_attr=dic["edge_attr"],
                        y = y)
        elif self.task == "test_local":
            act_index = self.train_num + self.val_num + index
            dic = torch.load(self.data_path + str(act_index) + ".pt")
            label_df = pd.read_csv("/home/jliao/webshell/tctrain/train.csv")
            y = 0 if label_df["label"][act_index] == "white" else 1
            return Data(num_nodes=dic["node_type"].shape[0], 
                        node_type= dic["node_type"], 
                        node_value = dic["node_value"],
                        edge_index = dic["edge_index"],
                        edge_attr=dic["edge_attr"],
                        y = y)

        elif self.task == "test": 
            dic = torch.load(self.data_path + str(index) + ".pt")
            return  Data(num_nodes=dic["node_type"].shape[0], 
                        node_type= dic["node_type"], 
                        node_value = dic["node_value"],
                        edge_index = dic["edge_index"],
                        edge_attr=dic["edge_attr"])
        
    def __len__(self):
        return self.data_len
