# -*- coding: utf-8 -*-
# @Time    : 2020/10/27 11:10
# @Author  : HH
# @File    : ConnectionGene.py
# @Project : ConvTest
import math
from NeatNode import Node
import numpy as np


class ConnectionGene:
    def __init__(self, fromn: Node, ton: Node, w: float, inno: int):
        self.fromNode = fromn  # type: Node
        self.toNode = ton  # type: Node
        self.weight = w
        self.innovationNo = inno
        self.enabled = True

    def isenabled(self) -> bool:
        return self.enabled

    def mutateweight(self):
        rand2 = np.random.rand()
        if rand2 < 0.1:
            self.weight = np.random.uniform(-1, 1)
        else:
            self.weight += np.random.normal(0, 1)/50
            if self.weight > 1:
                self.weight = 1
            elif self.weight < -1:
                self.weight = -1

    def clone(self, fromn: Node, ton: Node):
        cloner = ConnectionGene(fromn, ton, self.weight, self.innovationNo)
        cloner.enabled = True
        return cloner



