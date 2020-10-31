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
from matplotlib import pyplot as plt
changemap = dict()
active_name = "sigmoid"  # 激活函数名
nextConnectionNo = 1000
inputNo = 10
outputNo = 1
dataRatio = 15
eachPopNum = 500
seed = 2233
datasplit = 50
dataPath = "./datafile/final_data.csv"
breaktrainmark = True
hangupmark = False


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
    train_data = np.array(train_data)#1604124439.585793
    label_data = np.array(label_data)
    train_data, label_data = shuffle(train_data, label_data)
    train_idx, val_idx = next(
        iter(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(train_data, label_data)))
    X_train_train = train_data[train_idx]
    y_train_train = label_data[train_idx]
    X_train_val = train_data[val_idx]
    y_train_val = label_data[val_idx]
    return X_train_train, y_train_train, X_train_val, y_train_val


def breakthread():
    global breaktrainmark
    global hangupmark
    while(breaktrainmark):
        databreak = input("break thread started, key to control\n")
        if databreak == "s":
            breaktrainmark = False
        elif databreak == "h":
            if hangupmark:
                hangupmark = False
            else:
                hangupmark = True
        else:
            print("put again!")


def cutplot(nodeplot, layers):
    layer = [[] for x in range(layers)]
    global changemap
    layermaxnode = 0
    for node in nodeplot:
        layer[node[1]].append(node)
    for nodes in layer:
        if len(nodes) > layermaxnode:
            layermaxnode = len(nodes)
    for nodes in layer:
        for i in range(len(nodes)):
            if len(nodes) == layermaxnode:
                changemap[nodes[i][0]] = (i ) * layermaxnode / len(nodes)
            elif len(nodes)%2 == 1:
                changemap[nodes[i][0]] = (i+1) *layermaxnode /(len(nodes) + 1)
            else:
                changemap[nodes[i][0]] = (i + 1) * layermaxnode / (len(nodes) + 2)


if __name__ == "__main__":
    brakt = Thread(target=breakthread, args=())
    brakt.start()
    # f_log = open("log_file.txt", "w+")
    X_train_train, y_train_train, X_train_val, y_train_val = getdata()
    pop = Populations(eachPopNum)
    while True:
        if not breaktrainmark:
            break
        for i in range(1, int((len(X_train_train) / datasplit))):
            if not breaktrainmark:
                break
            index1 = i * datasplit
            index2 = (i + 1) * datasplit
            t1 = time.perf_counter()
            pop.checkloop(X_train_train[index1:index2], y_train_train[index1:index2])
            pop.naturalSelection()
            t2 = time.perf_counter()
            if hangupmark:
                nodeplot, connectplot, layers = pop.bestplayer.brain.printNodeMap()
                gap = 5
                cutplot(nodeplot, layers)
                plt.title("Neat Map")
                for node in nodeplot:
                    plt.plot(node[1] * gap, changemap.get(node[0]), "og")
                for connect in connectplot:
                    plt.plot([connect[0][1] * gap, connect[1][1] * gap],
                             [changemap.get(connect[0][0]), changemap.get(connect[1][0])], "-b")
                plt.show()
                hangupmark = False

            # f_log.write("fitness:{}, popnum:{} ,species:{}, gen:{}, rightnum/total:{}, rightnum:{}, timecost:{}, marktime:{}\n".format(pop.bestfitness, len(pop.pop), len(pop.species), pop.gen, pop.bestplayer.rightnum / datasplit, pop.bestplayer.rightnum, t2 - t1, time.time()))
            print("fitness:{}, popnum:{} ,species:{}, gen:{}, rightnum/total:{}, rightnum:{}, timecost:{}, marktime:{}".format(pop.bestfitness, len(pop.pop), len(pop.species), pop.gen, pop.bestplayer.rightnum / datasplit, pop.bestplayer.rightnum, t2 - t1, time.time()))
    # f_log.close()
