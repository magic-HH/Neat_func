# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:41
# @Author  : HH
# @File    : Genome.py
# @Project : ConvTest
from functools import singledispatch
from ConnectionHistory import *
from StartNet import *
from typing import *
from ConnectionGene import *
from NeatNode import *


class Genome:
    """"
        基因类，包含节点基因及连接基因
    """
    def __init__(self, layer_in=None, layer_out=None, crossover=None):
        self.genes = []  # type: list[ConnectionGene]
        self.nodes = []  # type: list[Node]
        self.network = []  # type: list[Node]
        self.inputs = 0
        self.outputs = 0
        self.layers = 2
        self.nextNode = 0
        self.biasNode = 0
        if crossover:
            self.inputs = layer_in
            self.outputs = layer_out
        elif layer_out and layer_in is not None:
            localNextConnectionNumber = 0
            self.inputs = layer_in
            self.outputs = layer_out
            for i in range(self.inputs):
                self.nodes.append(Node(i))
                self.nextNode += 1
                self.nodes[i].layer = 0
            for i in range(self.outputs):
                self.nodes.append(Node(i + self.inputs))
                self.nodes[i + self.inputs].layer = 1
                self.nextNode += 1
            self.nodes.append(Node(self.nextNode))  # 添加偏置节点
            self.biasNode = self.nextNode
            self.nextNode += 1
            self.nodes[self.biasNode].layer = 0
            for i in range(self.inputs):
                for j in range(self.outputs):
                    self.genes.append(
                        ConnectionGene(self.nodes[i], self.nodes[self.inputs + j], np.random.uniform(-1, 1),
                                       localNextConnectionNumber))
                    localNextConnectionNumber += 1
            for i in range(self.outputs):
                self.genes.append(
                    ConnectionGene(self.nodes[self.biasNode], self.nodes[self.inputs + i], np.random.uniform(-1, 1),
                                   localNextConnectionNumber))
                localNextConnectionNumber += 1

    def printNodeMap(self):
        nodeplot = []
        for tempnode in self.nodes:
            nodeplot.append([tempnode.number, tempnode.layer])
        connectplot = []
        for tempnodec in self.genes:
            connectplot.append([[tempnodec.fromNode.number, tempnodec.fromNode.layer], [tempnodec.toNode.number, tempnodec.toNode.layer], tempnodec.weight])
        return nodeplot, connectplot, self.layers

    def getNode(self, nodeNumber) -> bool:  # 获取节点
        for i in range(len(self.nodes)):
            if self.nodes[i].number == nodeNumber:
                return self.nodes[i]
        return None

    def connectNodes(self):  # 连接节点
        for i in range(len(self.nodes)):
            self.nodes[i].outputconnections.clear()
        for i in range(len(self.genes)):
            self.genes[i].fromNode.outputconnections.append(self.genes[i])

    def feedForward(self, inputvalues: list) -> List:  # 前向传播
        for i in range(self.inputs):
            self.nodes[i].outputvalue = inputvalues[i]
        self.nodes[self.biasNode].outputvalue = 1
        for i in range(len(self.network)):
            self.network[i].engage()

        outs = list(range(self.outputs))
        for i in range(self.outputs):
            outs[i] = self.nodes[i].outputvalue
        for i in range(len(self.nodes)):
            self.nodes[i].inputsum = 0
        return outs

    def generateNetwork(self):  # 生成神经网络结构
        self.connectNodes()
        self.network = []  # type: list[Node]
        for i in range(self.layers):
            for j in range(len(self.nodes)):
                if self.nodes[j].layer == i:
                    self.network.append(self.nodes[j])

    def getInnovationNumber(self, innovationhistory, fromnode, tonode) -> int:
        isNew = True
        global nextConnectionNo
        connectioninnovationnumber = nextConnectionNo
        for i in range(len(innovationhistory)):
            if innovationhistory[i].matches(self, fromnode, tonode):
                isNew = False
                connectioninnovationnumber = innovationhistory[i].innovationNumber
                break
        if isNew:
            innonumbers = []
            for i in range(len(self.genes)):
                innonumbers.append(self.genes[i].innovationNo)
            innovationhistory.append(ConnectionHistory(fromnode.number, tonode.number, connectioninnovationnumber, innonumbers))
            nextConnectionNo += 1
        return connectioninnovationnumber

    def addNode(self, innovationhistiry):  # 添加节点，用于变异
        randomconnection = np.random.randint(0, len(self.genes))
        while self.genes[randomconnection].fromNode == self.nodes[self.biasNode]:
            randomconnection = np.random.randint(0, len(self.genes))
        self.genes[randomconnection].enabled = False  # 禁用连接基因
        newNodeno = self.nextNode
        self.nodes.append(Node(newNodeno))
        self.nextNode += 1
        connectioninnovationnumber = self.getInnovationNumber(innovationhistiry, self.genes[randomconnection].fromNode,
                                                              self.getNode(newNodeno))
        self.genes.append(ConnectionGene(self.genes[randomconnection].fromNode, self.getNode(newNodeno), 1,
                                         connectioninnovationnumber))
        connectioninnovationnumber = self.getInnovationNumber(innovationhistiry, self.getNode(newNodeno),
                                                              self.genes[randomconnection].toNode)
        self.genes.append(ConnectionGene(self.getNode(newNodeno), self.genes[randomconnection].toNode, self.genes[randomconnection].weight, connectioninnovationnumber))
        self.getNode(newNodeno).layer = self.genes[randomconnection].fromNode.layer+1
        connectioninnovationnumber = self.getInnovationNumber(innovationhistiry, self.nodes[self.biasNode], self.getNode(newNodeno))
        self.genes.append(ConnectionGene(self.nodes[self.biasNode], self.getNode(newNodeno), 0, connectioninnovationnumber))
        if self.getNode(newNodeno).layer == self.genes[randomconnection].toNode.layer:
            for i in range(len(self.nodes) - 1):
                if self.nodes[i].layer >= self.getNode(newNodeno).layer:
                    self.nodes[i].layer = self.nodes[i].layer + 1
            self.layers += 1
        randomnodes = np.random.randint(0, len(self.nodes))
        while self.nodes[randomnodes].layer <= self.getNode(newNodeno).layer:
            randomnodes = np.random.randint(0, len(self.nodes))
        connectioninnovationnumber = self.getInnovationNumber(innovationhistiry, self.getNode(newNodeno), self.nodes[randomnodes])
        self.genes.append(ConnectionGene(self.getNode(newNodeno), self.nodes[randomnodes], 1, connectioninnovationnumber))

        self.connectNodes()

    def fullyConnection(self) -> bool:  # 检查是否全连接
        maxConnection = 0
        nodesinlayer = [0 for x in range(self.layers)]
        # print(self.nodes[i].layer)
        # print(nodesinlayer)
        # for i in range(len(self.nodes)):
        #     if self.nodes[i].layer >= self.layers:
        #         print("nodelayer above gen layer:")
        #         print(self.nodes[i].layer, self.layers)
        #         return False
        for i in range(len(self.nodes)):
            nodesinlayer[self.nodes[i].layer] += 1
        for i in range(self.layers - 1):
            nodesinfront = 0
            for j in range(i + 1, self.layers):
                nodesinfront += nodesinlayer[j]
            maxConnection += nodesinlayer[i] * nodesinfront
        if maxConnection == len(self.genes):
            return True
        return False

    def mutate(self, innovationhistory):  # 杂交时变异
        rand1 = np.random.uniform(0, 1)
        if rand1 < 0.8:
            for i in range(len(self.genes)):
                self.genes[i].mutateweight()
        rand2 = np.random.uniform(0, 1)
        if rand2 < 0.5:
            self.addConnection(innovationhistory)
        rand3 = np.random.uniform(0, 1)
        if rand3 < 0.03:
            self.addNode(innovationhistory)

    def addConnection(self, innovationhistory):  # 添加一条新的连接
        if self.fullyConnection():
            # print("connection failed!")
            return
        randomnode1 = np.random.randint(0, len(self.nodes))
        randomnode2 = np.random.randint(0, len(self.nodes))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                if (self.nodes[i].layer != self.nodes[j].layer) and (not (self.nodes[i].isconnectedto(self.nodes[j])) or not(self.nodes[j].isconnectedto(self.nodes[i]))):
                    randomnode1 = i
                    randomnode2 = j
                    break
        if self.nodes[randomnode1].layer > self.nodes[randomnode2].layer:
            temp = randomnode2
            randomnode2 = randomnode1
            randomnode1 = temp
        connectioninnovationnumber = self.getInnovationNumber(innovationhistory, self.nodes[randomnode1], self.nodes[randomnode2])
        self.genes.append(ConnectionGene(self.nodes[randomnode1], self.nodes[randomnode2], np.random.uniform(-1, 1), connectioninnovationnumber))
        self.connectNodes()

    def crossover(self, parent2):  # 杂交
        child = Genome(self.inputs, self.outputs, True)
        child.genes.clear()
        child.nodes.clear()
        child.layers = self.layers
        child.nextNode = self.nextNode
        child.biasNode = self.biasNode
        childgens = []  # type: list[ConnectionGene]
        isenabled = []  # type: list[bool]
        for i in range(len(self.genes)):
            setenabled = True
            parent2gene = self.matchingGene(parent2, self.genes[i].innovationNo)
            if parent2gene != -1:
                if (not self.genes[i].enabled) or (not parent2.genes[parent2gene].enabled):
                    if np.random.uniform(0, 1) < 0.75:
                        setenabled = False
                rand = np.random.uniform(0, 1)
                if rand < 0.5:
                    childgens.append(self.genes[i])
                else:
                    childgens.append(parent2.genes[parent2gene])
            else:
                childgens.append(self.genes[i])
                setenabled = self.genes[i].enabled
            isenabled.append(setenabled)
        for i in range(len(self.nodes)):
            child.nodes.append(self.nodes[i].clone())
        for i in range(len(childgens)):
            child.genes.append(childgens[i].clone(child.getNode(childgens[i].fromNode.number), child.getNode(childgens[i].toNode.number)))
            child.genes[i].enabled = isenabled[i]
        child.connectNodes()
        return child

    def matchingGene(self, parent2, innovationnumber: int) -> int:
        for i in range(len(parent2.genes)):
            if parent2.genes[i].innovationNo == innovationnumber:
                return i
        return -1

    def clone(self):  # 克隆
        cloner = Genome(self.inputs, self.outputs, True)
        for i in range(len(self.nodes)):
            cloner.nodes.append(self.nodes[i].clone())
        for i in range(len(self.genes)):
            cloner.genes.append(self.genes[i].clone(cloner.getNode(self.genes[i].fromNode.number),
                                                    cloner.getNode(self.genes[i].toNode.number)))
        cloner.layers = self.layers
        cloner.nextNode = self.nextNode
        cloner.biasNode = self.biasNode
        cloner.connectNodes()
        return cloner
    """
        未实现功能：绘制Neat map
    
    """