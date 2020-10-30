# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 10:36
# @Author  : HH
# @File    : ConnectionHistory.py
# @Project : ConvTest
from ConnectionGene import *
from NeatNode import *
from Genome import *


class ConnectionHistory:
    def __init__(self, fromn: int, ton: int, inno: int, innovationnos: list):
        self.fromNode = fromn
        self.toNode = ton
        self.innovationNumber = inno
        self.innovationNumbers = innovationnos.copy()

    def matches(self, genome, fromn: Node, ton: Node) -> bool:
        if len(genome.genes) == len(self.innovationNumbers):
            if fromn.number == self.fromNode and ton.number == self.toNode:
                for i in range(len(genome.genes)):
                    if genome.genes[i].innovationNo in self.innovationNumbers:
                        return False
                return True
        return True
