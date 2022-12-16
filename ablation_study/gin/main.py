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
from utils import load_pickle,setup_seed,save_csv_with_configname,bestGPU
import copy
import time

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
    for _, batch in enumerate(tqdm(loader, desc="Iteration",mininterval=50)):
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
    # print("acc:", acc, "recall:", recall, "precision:", precision)
    # print("score:", score)
    result = {
        "acc" : acc,
        "recall": recall,
        "precision": precision,
        "score": score,
    }
    return score, result

def eval_test(model,device,loader):
    model.eval()
    y_pred = []
    for _, batch in enumerate(tqdm(loader, desc="Iteration",mininterval=50)):
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

def eval_per_sample(model, device, loader, net_parameters):
    model.eval()
    model_state_dict, _ = load_best_epoch(net_parameters["model_path"])
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        pred = model(batch.node_type,batch.node_value,batch.edge_index,batch.edge_attr,batch.batch)


def main(net_parameters):
    setup_seed(net_parameters["seed"], torch.cuda.is_available())
    device = torch.device(f"cuda:{bestGPU(True)}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(net_parameters["model_path"]):
        os.makedirs(net_parameters["model_path"])
    
    total= pd.read_csv('/home/jliao/webshell/tctrain/train.csv')
    total_num  = total.shape[0]
    
    vocab_dir = "vocab/"
    
    train_dir = '/home/jliao/webshell/tctrain/train/'
    train_dataset_dir = f"split{net_parameters['split_type']}/train_dataset/"
    val_dataset_dir = f"split{net_parameters['split_type']}/val_dataset/"
    test_dataset_dir = f"split{net_parameters['split_type']}/test_dataset/"

    if net_parameters["split_type"] == 1:
        train_num = int(4/6*total_num)
        val_num = int(1/6*total_num)
        test_num = total_num - train_num - val_num
        print("train/val/test_num:", train_num," ",val_num," ",test_num)
    else:
        # split2:4:1:50
        train_num = int(4/55*total_num)
        val_num = int(1/55*total_num)
        test_num = total_num - train_num - val_num
        print("train/val/test_num:", train_num," ",val_num," ",test_num)


    if net_parameters["extract"] == True and not os.path.exists(train_dataset_dir):
        print("Start building vocabulary on train dataset...")
        extract_vocabulary(train_num, train_dir, 10000,2,vocab_dir)
        type_word2id  = load_pickle(vocab_dir + "word2id.node_types.pkl")
        value_word2id = load_pickle(vocab_dir + "word2id.node_values.pkl")
        extract_and_save_graph_data( 1,train_num+1, train_dir,train_dataset_dir,type_word2id,value_word2id)
        extract_and_save_graph_data(train_num+1,train_num+val_num+ 1, train_dir,val_dataset_dir,type_word2id,value_word2id)
        torch.cuda.synchronize()
        extract_start = time.time()
        extract_and_save_graph_data(train_num+val_num+ 1,total_num+1, train_dir,test_dataset_dir,type_word2id,value_word2id)
        torch.cuda.synchronize()
        extract_end= time.time()
        extract_time = (extract_end-extract_start)/(total_num-train_num-val_num)
        print(f"extract per test sample time: {extract_time} sec")
    else: 
        type_word2id  = load_pickle(vocab_dir + "word2id.node_types.pkl")
        value_word2id = load_pickle(vocab_dir + "word2id.node_values.pkl")
        extract_time = 0

    net_parameters["type_nums"] = len(type_word2id)
    net_parameters["value_nums"] = len(value_word2id)

    load_train_dataset = get_load_dataset(train_dataset_dir,"train",train_num,val_num) 
    load_val_dataset = get_load_dataset(val_dataset_dir,"val",train_num,val_num)
    load_test_local_dataset =  get_load_dataset(test_dataset_dir,"test_local",train_num,val_num)

    train_loader = DataLoader(load_train_dataset(), batch_size = net_parameters["batch_size"], shuffle = True)
    val_loader = DataLoader(load_val_dataset(), batch_size = net_parameters["batch_size"], shuffle = True)
    test_local_loader = DataLoader(load_test_local_dataset(), batch_size = net_parameters["batch_size"], shuffle = True)
    test_per_local_loader = DataLoader(load_test_local_dataset(), batch_size = 1, shuffle = True)

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
    continues_fials = net_parameters["continues_fials"]
    torch.cuda.synchronize()
    train_start = time.time()
    for epoch in range(net_parameters["load"] + 1, net_parameters["epochs"] + 1):
        print("=====Epoch {} ====".format(epoch))
        print("Training...")
        losses = []
        model.train()
        for _, batch in enumerate(tqdm(train_loader,desc="Iteration",mininterval=50)):
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
            print("Evaluating the validation dataset...")
            # eval(model,device,train_loader)
            cur_acc, _ = eval(model,device,val_loader)
            if cur_acc > best_valid_acc:
                best_valid_acc = cur_acc
                torch.save(
                    (model.state_dict(), optimizer.state_dict()),
                    os.path.join(net_parameters["model_path"], "best_validation.pth"),
                )
            continues_fials=net_parameters["continues_fials"]
        else:
            continues_fials=continues_fials-1
            if continues_fials==0:
                print(f"The perfo rmance of the model has not been improved by consecutive {net_parameters['continues_fials']} epoch, early stop")
                break
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    train_end = time.time()
    train_time = train_end-train_start
    print(f"total training time: {train_time} sec")
    ##### test local #####
    torch.cuda.synchronize()
    test_start = time.time()
    model.eval()
    model_state_dict, _ = load_best_epoch(net_parameters["model_path"])
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    score, result = eval(model,device,test_local_loader)
    torch.cuda.synchronize()
    test_end = time.time()
    test_time = test_end - test_start
    print(f"total testing time: {test_time} sec")
    print("test local result:")
    print(result)
    torch.cuda.synchronize()
    per_test_start = time.time()
    eval_per_sample(model, device, test_per_local_loader, net_parameters)
    torch.cuda.synchronize()
    per_test_end = time.time()
    per_test_time = per_test_end - per_test_start
    print(f"per test sample testing time: {per_test_time} sec")
    return score, result, train_time, test_time, per_test_time, extract_time


    """
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
    """

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--config", type= str ,help =" the dir of config file"
#     )
#     args = parser.parse_args()
#     config = args.config
#     config_name = config.split("/")[-1].split(".")[0]
#     with open(config) as f:
#         total_parameters = json.load(f)
#     if isinstance(total_parameters["seed"], list):
#         for seed in total_parameters["seed"]:
#             if isinstance(total_parameters["JK"],list):
#                 for jk in total_parameters["JK"]:
#                     if isinstance(total_parameters["pooling"], list):
#                         for pooling in total_parameters["pooling"]:
#                             print("current seed: ", seed, "jk:", jk, "pooling:", pooling)
#                             net_parameters = copy.deepcopy(total_parameters)
#                             net_parameters["seed"] = seed
#                             net_parameters["JK"] = jk
#                             net_parameters["pooling"] = pooling
#                             net_parameters["model_path"] = (
#                                 net_parameters["model_path"] + "/" + str(net_parameters["seed"])
#                             )
#                             result = main(net_parameters)
#                             net_parameters.update(result)
#                             save_csv_with_configname(net_parameters, "result", config_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type= str,default="configs/gin/split2/config_jk2_pooling3.json" ,help =" the dir of config file"
    )
    parser.add_argument(
        "--num_layer", type= str,default=None ,help =" the nums of layers"
    )
    args = parser.parse_args()
    config = args.config
    config_name = config.split("/")[-1].split(".")[0]
    with open(config) as f:
        total_parameters = json.load(f)
    if args.num_layer is not None:
        total_parameters['num_layer']=args.num_layer
    if isinstance(total_parameters["seed"], list):
        results_list = {}
        for seed in total_parameters["seed"]:
            print("current seed: ", seed)
            net_parameters = copy.deepcopy(total_parameters)
            net_parameters["seed"] = seed
            net_parameters["model_path"] = (
                net_parameters["model_path"] + "/" + str(net_parameters["seed"])
            )
            score, result, train_time, test_time, per_test_time, extract_time = main(net_parameters)
            # for k in result.keys():
            #     if k not in results_list.keys():
            #         results_list[k] = []
            #     results_list[k].append(result[k]*100)
            if 'score' not in results_list.keys():
                results_list['score'] = []
            results_list['score'].append(score)
            if 'train_time' not in results_list.keys():
                results_list['train_time'] = []
            results_list['train_time'].append(train_time)
            if 'test_time' not in results_list.keys():
                results_list['test_time'] = []
            results_list['test_time'].append(test_time)
            if 'per_test_time' not in results_list.keys():
                results_list['per_test_time'] = []
            results_list['per_test_time'].append(per_test_time)
            if 'extract_time' not in results_list.keys():
                results_list['extract_time'] = []
            results_list['extract_time'].append(extract_time)
            net_parameters.update(result)
            save_csv_with_configname(net_parameters, "result", config_name)
        # print(f"Avg. {np.mean(np.array(score_list))} +- {np.std(np.array(score_list))*100:.2f}%")
        # fout = open(f"configs/{net_parameters['gnn_type']}/split{net_parameters['split_type']}/layers_rslt.txt", 'a')
        # fout.write(f"{args.config.split('/')[-1][:-6]},{np.mean(np.array(score_list))}, Avg. {np.mean(np.array(score_list))} +- {np.std(np.array(score_list))*100:.2f}%\n")
        fout = open(f"{args.config[:args.config.rfind('/')]}/{net_parameters['gnn_type']}_split{net_parameters['split_type']}_rslt.txt", 'a')
        for k,v in results_list.items():
            fout.write(f"Avg.{k}:{np.mean(np.array(v))}+-{np.std(np.array(v)):.2f}%, ")
        fout.write('\n')
    elif isinstance(total_parameters["seed"], int):
        # score_list = []
        # for seed in total_parameters["seed"]:
        seed=total_parameters["seed"]
        print("current seed: ", seed)
        net_parameters = copy.deepcopy(total_parameters)
        net_parameters["seed"] = seed
        net_parameters["model_path"] = (
            net_parameters["model_path"] + "/" + str(net_parameters["seed"])
        )
        score, result, train_time, test_time, extract_time = main(net_parameters)
            # score_list.append(score)
        net_parameters.update(result)
        save_csv_with_configname(net_parameters, "result", config_name)
        # print(f"Avg. {np.mean(np.array(score_list))} +- {np.std(np.array(score_list))*100:.2f}%")
        # fout = open(f"configs/{net_parameters['gnn_type']}/split{net_parameters['split_type']}/rslt.txt", 'a')
        # # fout.write(f"{args.config.split('/')[-1][:-6]},{np.mean(np.array(score_list))}, Avg. {np.mean(np.array(score_list))} +- {np.std(np.array(score_list))*100:.2f}%\n")
        # fout.write(f"{args.config.split('/')[-1][:-5]}, Avg.{np.mean(np.array(score_list))} +- {np.std(np.array(score_list))*100:.2f}%\n")