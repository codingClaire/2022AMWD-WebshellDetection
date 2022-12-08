import pickle
import torch
import random
import numpy as np
import os
import csv

def save_pickle(obj, path):

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def setup_seed(seed, is_cuda):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # import random
    random.seed(seed)

def save_csv_with_configname(params,file_dir,configname):
    # 1. change dics
    print(params)
    pool_method = params["pooling"]

    # update new parameters
    if(pool_method in params.keys()):
        for pool_param in params[pool_method].keys():
            new_key = pool_method +"_" + pool_param
            params[new_key] = params[pool_method][pool_param]

        del params[pool_method]
    print(params)
    
    sheet_title = params.keys()
    sheet_title = sorted(sheet_title)
    print(sheet_title)
    sheet_data = []
    for key in sheet_title:
        sheet_data.append(params[key])
    # 2. find the file
    file_name = pool_method+"_["+configname+"].csv"
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(os.path.join(file_dir, file_name)):
        action = "a"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
    else:
        action = "w"
        csv_fp = open(os.path.join(file_dir, file_name),
                      action, encoding='utf-8', newline='')
        writer = csv.writer(csv_fp)
        writer.writerow(sheet_title)
    writer.writerow(sheet_data)
    csv_fp.close()
def bestGPU(gpu_verbose=False, **w):
    import GPUtil
    import numpy as np

    Gpus = GPUtil.getGPUs()
    Ngpu = 4
    mems, loads = [], []
    for ig, gpu in enumerate(Gpus):
        memUtil = gpu.memoryUtil * 100
        load = gpu.load * 100
        mems.append(memUtil)
        loads.append(load)
        if gpu_verbose: print(f'gpu-{ig}:   Memory: {memUtil:.2f}%   |   load: {load:.2f}% ')
    bestMem = np.argmin(mems)
    bestLoad = np.argmin(loads)
    best = bestMem
    if gpu_verbose: print(f'//////   Will Use GPU - {best}  //////')

    return int(best)