import torch
import torch.optim as optim

import pandas as pd
import csv
import os
import numpy as np
from tqdm import tqdm
import json
import argparse
from itertools import chain

from extract_data import extract_and_save_graph_data,extract_vocabulary,extract_test_graph_data
from torch_geometric.data import DataLoader
from dataloader import get_load_dataset
from model import predictModel
from utils import load_pickle

bcls_criterion = torch.nn.BCEWithLogitsLoss()

def load_best_epoch(model_path):
    print('loading best epoch')
    return torch.load(os.path.join(model_path, 'best_validation.pth'),
                      map_location='cpu')

def load_epoch(model_path, epoch):
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(model_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')

def eval(model,device,loader):
    model.eval()
    y_pred = []
    TP,FN,FP,TN = 0,0,0,0
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.node_type,batch.node_value,batch.edge_index,batch.edge_attr,batch.batch)
        output = torch.sigmoid(pred.detach()).view(-1, 1).cpu()
        y_pred.append(
            torch.where(
                output > 0.5, torch.ones_like(output), torch.zeros_like(output)
            ).to(device)
        )
        # total_y +=len(y_pred[-1])
        for i in range(len(y_pred[-1])):
            if(batch.y[i] == 1 and y_pred[-1][i] == 1):
                TP+=1
            elif(batch.y[i] == 1 and y_pred[-1][i] == 0):
                FN+=1
            elif(batch.y[i] == 0 and y_pred[-1][i] == 1):
                FP+=1
            elif(batch.y[i] == 0 and y_pred[-1][i] == 0):
                TN+=1
    acc = (TP+TN) / (TP+TN+FN+FP)
    recall = (TP) / (TP + FN)
    precision = TP / (TP+FP)
    score = (1+0.5*0.5) *(precision * recall)/((0.5*0.5*precision) + recall)
    print("acc:", acc, "recall:", recall, "precision:", precision)
    print("score:", score)
    return score

def eval_test(model,device,loader):
    model.eval()
    y_pred = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.node_type,batch.node_value,batch.edge_index,batch.edge_attr,batch.batch)
        output = torch.sigmoid(pred.detach()).view(-1, 1).cpu()
        y_pred.append(
            torch.where(
                output > 0.5, torch.ones_like(output), torch.zeros_like(output)
            ).to(device)
        )
    return list(chain(*y_pred))
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type= str ,help =" the dir of config file",default='config.json'
    )
    args = parser.parse_args()
    config = args.config
    with open(config) as f:
        net_parameters = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(net_parameters["model_path"]):
        os.makedirs(net_parameters["model_path"])
    
    total= pd.read_csv('tctrain/train.csv')
    # total_labels = total["label"].tolist()
    total_num  = total.shape[0]
    """
    for i in range(0,total_num):
        if (total_labels[i] =="white"):
            total_labels[i] = 0
        else:
            total_labels[i] = 1
    """
    
    vocab_dir = "vocab/"
    
    train_dir = 'tctrain/train/'
    train_dataset_dir = "train_dataset/"
    val_dataset_dir = "val_dataset/"

    train_num = int(0.8*total_num)

    print("train_num:", train_num)
    print("val_num:", total_num-train_num)

    if net_parameters["extract"] == True:
        print("Start building vocabulary on train dataset...")
        extract_vocabulary(train_num, train_dir, 10000,2,vocab_dir)
        type_word2id  = load_pickle(vocab_dir + "word2id.node_types.pkl")
        value_word2id = load_pickle(vocab_dir + "word2id.node_values.pkl")
        extract_and_save_graph_data(1,train_num+1, train_dir,train_dataset_dir,type_word2id,value_word2id)
        extract_and_save_graph_data(train_num+1,total_num+1, train_dir,val_dataset_dir,type_word2id,value_word2id)
    else: 
        type_word2id  = load_pickle(vocab_dir + "word2id.node_types.pkl")
        value_word2id = load_pickle(vocab_dir + "word2id.node_values.pkl")
        
    net_parameters["type_nums"] = len(type_word2id)
    net_parameters["value_nums"] = len(value_word2id)

    load_train_dataset = get_load_dataset(train_dataset_dir,"train",train_num) # todo
    load_val_dataset = get_load_dataset(val_dataset_dir,"val",train_num)
    train_loader = DataLoader(load_train_dataset(), batch_size = net_parameters["batch_size"], shuffle = True)
    val_loader = DataLoader(load_val_dataset(), batch_size = net_parameters["batch_size"], shuffle = True)


    model = predictModel(net_parameters).to(device)
    
    optimizer_state_dict = None
    if net_parameters["load"] > 0:
        model_state_dict, optimizer_state_dict = load_epoch(net_parameters["model_path"], net_parameters["load"])
        model.load_state_dict(model_state_dict)    
    
    
    optimizer = optim.Adam(
        model.parameters(), lr=net_parameters["learning_rate"]
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    loss_list = []
    best_valid_acc = 0
    continues_fials=net_parameters['continues_fials']
    for epoch in range(net_parameters["load"] + 1, net_parameters["epochs"] + 1):
        print("=====Epoch {} ====".format(epoch))
        print("Training...")
        losses = []
        for _, batch in enumerate(tqdm(train_loader)):
            batch = batch.to(device)
            pred = model(batch.node_type,batch.node_value,batch.edge_index,batch.edge_attr,batch.batch)
            optimizer.zero_grad()
            is_labeled = batch.y == batch.y
            if len(batch.y.shape) == 1:
                batch.y = torch.unsqueeze(batch.y, 1)
            loss = bcls_criterion(
                        pred.to(torch.float32)[is_labeled],
                        batch.y.to(torch.float32)[is_labeled],
                    )
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if (net_parameters["save_every_epoch"] and epoch % net_parameters["save_every_epoch"] == 0):
            tqdm.write("saving to epoch.%04d.pth" % epoch)
            torch.save(
                (model.state_dict(), optimizer.state_dict()),
                os.path.join(net_parameters["model_path"], "epoch.%04d.pth" % epoch),
            )
        loss = np.mean(losses)
        loss_list.append(loss)
        print(loss_list)
        
        if net_parameters["eval"] == True:
            print("Evaluating the training dataset...")
            # eval(model,device,train_loader)
            cur_acc = eval(model,device,val_loader)
            if cur_acc > best_valid_acc:
                print(f"best acc update, best acc={cur_acc}")
                best_valid_acc = cur_acc
                torch.save(
                        (model.state_dict(), optimizer.state_dict()),
                        os.path.join(net_parameters["model_path"], "best_validation.pth"),
                    )
                continues_fials=net_parameters['continues_fials']
                
            else:
                continues_fials-=1
                if continues_fials==0:
                    print(f"The performance of the model has not been improved by consecutive {net_parameters['continues_fials']} epoch, early stop")
                    break
    ##### test #####
    test_dir = 'tcdata/test/'
    test_data_dir = "test_dataset/"

    test= pd.read_csv('tcdata/test.csv')
    test_id = test["file_id"].tolist()
    _,realid2fileid = extract_test_graph_data(test_id,test_dir, test_data_dir,type_word2id, value_word2id)
    
    load_test_dataset = get_load_dataset(test_data_dir,"test",train_num)
    test_loader = DataLoader(load_test_dataset(),batch_size = net_parameters["batch_size"])
    model_state_dict, _ = load_best_epoch(net_parameters["model_path"])
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    test_result = eval_test(model,device,test_loader)

    with open("result.csv","w") as out:
        f = csv.writer(out)
        f.writerow(("file_id","label"))
        for i in test_id:
            file_id = realid2fileid[i]
            label = "white" if test_result[file_id].item() == 0 else "black"
            f.writerow((i, label))

if __name__ == "__main__":
    main()