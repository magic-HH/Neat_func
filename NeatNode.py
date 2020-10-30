# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 10:28
# @Author  : HH
# @File    : NeatNode.py
# @Project : ConvTest
from StartNet import *
from ActiveFunc import func_active
from ConnectionGene import *


class Node:
    def __init__(self, no=0):
        self.number = no
        self.inputsum = 0
        self.outputvalue = 0
        self.outputconnections = []  # type: list[ConnectionGene]
        self.layer = 0

    def engage(self):
        if self.layer != 0:
            self.outputvalue = self.activation(self.inputsum)
        for i in range(len(self.outputconnections)):
            self.outputconnections[i].toNode.inputsum += self.outputconnections[i].weight * self.outputvalue

    def activation(self, x: float) -> float:
        return func_active.get(active_name)(x)

    def stepfunction(self, x: float) -> int:
        if x < 0:
            return 0
        else:
            return 1

    def isconnectedto(self, node) -> bool:
        if node.layer == self.layer:
            return False
        if node.layer < self.layer:
            for i in range(len(node.outputconnections)):
                if node.outputconnections[i].toNode == self:
                    return True
        else:
            for i in range(len(self.outputconnections)):
                if self.outputconnections[i].toNode == node:
                    return True
        return False

    def clone(self):
        cloner = Node(self.number)
        cloner.layer = self.layer
        return cloner
