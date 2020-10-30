# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 10:48
# @Author  : HH
# @File    : StartNet.py
# @Project : ConvTest
import time
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
from Population import *
from threading import Thread

active_name = "sigmoid"  # 激活函数名
nextConnectionNo = 1000
inputNo = 10
outputNo = 1
dataRatio = 15
eachPopNum = 500
seed = 2233
datasplit = 80
dataPath = "./datafile/final_data.csv"
breaktrainmark = True


def timecount(func):
    def count(*args, **kwargs):
        t1 = time.perf_counter()
        func(*args, **kwargs)
        print(kwargs)
        print(*args)
        t2 = time.perf_counter()
        print("time cost:{}".format(t2 - t1))
    return count


def getdata():
    datas = pd.read_csv(dataPath, index_col=0)
    data_list = datas.values.tolist()
    train_data = []
    label_data = []
    for d in data_list:
        label_data.append(d.pop(-1))
        train_data.append(d)
    train_data = np.array(train_data)
    label_data = np.array(label_data)
    train_data, label_data = shuffle(train_data, label_data)
    train_idx, val_idx = next(
        iter(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(train_data, label_data)))
    X_train_train = train_data[train_idx]
    y_train_train = label_data[train_idx]
    X_train_val = train_data[val_idx]
    y_train_val = label_data[val_idx]
    return X_train_train, y_train_train, X_train_val, y_train_val


# def breakthread():
#     global breaktrainmark
#     databreak = input("break thread started, key 's' to get out\n")
#     if databreak == "s":
#         breaktrainmark = False


if __name__ == "__main__":
    # brakt = Thread(target=breakthread, args=())
    # brakt.start()
    f_log = open("log_file.txt")
    X_train_train, y_train_train, X_train_val, y_train_val = getdata()
    pop = Populations(eachPopNum)
    while True:
        if not breaktrainmark:
            break
        for i in range(1, int((len(X_train_train) / datasplit))):
            index1 = i * datasplit
            index2 = (i + 1) * datasplit
            t1 = time.perf_counter()
            pop.checkloop(X_train_train[index1:index2], y_train_train[index1:index2])
            pop.naturalSelection()
            t2 = time.perf_counter()
            f_log.write("fitness:{}, popnum:{} ,species:{}, gen:{}, rightnum/total:{}, rightnum:{}, timecost:{}, marktime:{}".format(pop.bestfitness, len(pop.pop), len(pop.species), pop.gen, pop.bestplayer.rightnum / datasplit, pop.bestplayer.rightnum, t2 - t1, time.time()))
            # print("fitness:{}, popnum:{} ,species:{}, gen:{}, rightnum/total:{}, rightnum:{}, timecost:{}".format(
            #     pop.bestfitness, len(pop.pop), len(pop.species), pop.gen, pop.bestplayer.rightnum / datasplit,
            #     pop.bestplayer.rightnum, t2 - t1))
