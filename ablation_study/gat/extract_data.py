import os
import json
import queue
from tqdm import tqdm
from collections import Counter
from utils import save_pickle
import numpy as np
import torch
from torch_geometric.utils import to_undirected

def extract_test_graph_data(real_ids, file_dir,output_dir,type_word2id,value_word2id):
    # extract test id
    fid2rid,rid2fid = {},{}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for num in tqdm(range(len(real_ids))):
        id = 0
        q = queue.Queue()
        dic = {}
        node_types,node_values = [],[]
        index1, index2 = [], []
        fid2rid[num] = real_ids[num]
        rid2fid[real_ids[num]] = num
        with open(os.path.join(file_dir+str(real_ids[num])),"r") as f:
            tree = json.load(f)
            q.put(tree)
            while not q.empty():
                subtree = q.get()
                if ":" in subtree["name"]:
                    subtree_name = subtree["name"][1:-1]
                    node_type = subtree_name.split(":")[0]
                    node_value = subtree_name.split(":")[1]
                    if node_value == "":
                        node_value = node_type
                    else:
                        node_value = node_value[1:-1]
                else:
                    node_type = subtree["name"]
                    node_value = subtree["name"]
                node_types.append(convert_word_to_id(node_type, type_word2id))
                node_values.append(convert_word_to_id(node_value, value_word2id))
                if "preid" in subtree:
                    index1.append(subtree["preid"])
                    index2.append(id)
                children_list = subtree["children"]
                clen = len(children_list)
                for i in range(clen):
                    children_list[i]["preid"] = id
                    q.put(children_list[i])
                id+=1
        dic["edge_index"] = to_undirected(torch.tensor(np.stack([index1,index2])))
        dic["edge_attr"] = torch.ones(torch.Size([len(index1),1]))
        dic["node_type"] = torch.tensor(node_types)
        dic["node_value"] = torch.tensor(node_values)
        torch.save(dic,output_dir + str(num) +".pt")

    return fid2rid,rid2fid

def extract_and_save_graph_data(undirected, start,end, file_dir,output_dir,type_word2id,value_word2id):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for num in tqdm(range(start,end)):
        id = 0
        q = queue.Queue()
        dic = {}
        node_types,node_values = [],[]
        index1, index2 = [], []
        with open(os.path.join(file_dir+str(num)),"r") as f:
            tree = json.load(f)
            q.put(tree)
            while not q.empty():
                subtree = q.get()
                if ":" in subtree["name"]:
                    subtree_name = subtree["name"][1:-1]
                    node_type = subtree_name.split(":")[0]
                    node_value = subtree_name.split(":")[1]
                    if node_value == "":
                        node_value = node_type
                    else:
                        node_value = node_value[1:-1]
                else:
                    node_type = subtree["name"]
                    node_value = subtree["name"]
                node_types.append(convert_word_to_id(node_type, type_word2id))
                node_values.append(convert_word_to_id(node_value, value_word2id))
                if "preid" in subtree:
                    index1.append(subtree["preid"])
                    index2.append(id)
                children_list = subtree["children"]
                clen = len(children_list)
                for i in range(clen):
                    children_list[i]["preid"] = id
                    q.put(children_list[i])
                id+=1
        if undirected:
            dic["edge_index"] = to_undirected(torch.tensor(np.stack([index1,index2])))
        else:
            dic["edge_index"] = torch.tensor(np.stack([index1,index2]))
        dic["edge_attr"] = torch.ones(torch.Size([len(index1),1]))
        dic["node_type"] = torch.tensor(node_types)
        dic["node_value"] = torch.tensor(node_values)
        torch.save(dic,output_dir + str(num-1) +".pt")


def build_vocab(texts,max_vocab_size,min_frequency= 2, reserved = None):
    if reserved is None:
        reserved = ['<PAD>', '<UNK>','<SLOT>','<BOS>','<EOS>']
    # count each words' appear times
    counter = Counter(texts)
    original_vocab_size = len(counter)
    # filt words that appear less than min_frequency
    counter_items = list(filter(
        lambda x: x[1] >= min_frequency and x[0] not in reserved,
        counter.items()))
    # sort counter from large to small frequency 
    counter_items.sort(key=lambda x: x[1], reverse=True)
    id2word = list(map(lambda x: x[0], counter_items))
    # must include reserved
    if max_vocab_size is None:
        id2word = reserved + id2word
    else:
        id2word = reserved + id2word[:max_vocab_size - len(reserved)]
    word2id={v: k for k, v in enumerate(id2word)}
    return (word2id,id2word, original_vocab_size)


def convert_word_to_id(word,word2id):
    unknown_code = word2id['<UNK>']
    return word2id.get(word, unknown_code)

def convert_words_to_ids(words, word2id):
    unknown_code = word2id['<UNK>']
    id_list= list(map(lambda x: word2id.get(x, unknown_code), words))
    return id_list


def extract_vocabulary(total, data_path,max_vocab_size, min_frequency,vocab_path):
    node_types, node_values= [], []
    for num in tqdm(range(1,total)):
        q = queue.Queue()
        with open(os.path.join(data_path+str(num)),"r") as f:
            tree = json.load(f)
            q.put(tree)
            while not q.empty():
                subtree = q.get()
                if ":" in subtree["name"]:
                    subtree_name = subtree["name"][1:-1]
                    node_type = subtree_name.split(":")[0]
                    node_value = subtree_name.split(":")[1]
                    if node_value == "":
                        node_value = node_type
                    else:
                        node_value = node_value[1:-1]
                else:
                    node_type = subtree["name"]
                    node_value = subtree["name"]
                node_types.append(node_type)
                node_values.append(node_value)
                children_list = subtree["children"]
                clen = len(children_list)
                for i in range(clen):
                    q.put(children_list[i])

    statistics = {}
    items = ["node_types", "node_values"]
    for item in items:
        if item == "node_types":
            word2id, id2word, original_vocab_size = build_vocab(
                node_types, max_vocab_size, min_frequency
            )
        elif item == "node_values":
            word2id, id2word, original_vocab_size = build_vocab(
                node_values, max_vocab_size, min_frequency
            )

        statistics["%sOriginalVocabSize" % item] = original_vocab_size
        statistics["%sVocabSize" % item] = len(id2word)
        # save word2id and id2word
        if not os.path.exists(vocab_path):
            os.makedirs(vocab_path)
        save_pickle(word2id, os.path.join(vocab_path, "word2id.%s.pkl" % item))
        save_pickle(id2word, os.path.join(vocab_path, "id2word.%s.pkl" % item))
    print(statistics)




