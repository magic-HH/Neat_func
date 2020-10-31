# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 18:54
# @Author  : HH
# @File    : FuncProcess.py
# @Project : ConvTest
import numpy as np
from StartNet import inputNo, outputNo, dataRatio
from Genome import *


class Player:
    def __init__(self):
        self.fitness = 1
        self.brain = Genome(inputNo, outputNo)
        self.inputdata = []
        self.decision = []
        self.answer = []
        self.totalnum = 0
        self.rightnum = 0

    def calculateFitness(self):  # fitnesscount\
        if self.answer is not None and self.decision is not None:
            self.totalnum = len(self.answer)
            self.rightnum = 0
            self.fitness = 1
            for i in range(self.totalnum):
                if self.decision[i][0] >= 0.5:
                    tempanswer = 1
                else:
                    tempanswer = 0
                if self.answer[i] == tempanswer:
                    self.rightnum += 1
                    if self.answer[i] == 0:
                        self.fitness *= np.random.uniform(2, dataRatio - 7)

            if self.rightnum == 0:
                self.rightnum = np.random.uniform(0, 0.3)

            self.fitness *= self.rightnum * np.random.uniform(1, 3)
            # if self.fitness >= 2:
            #     print("player fitness: {}".format(self.fitness))

    def pridectData(self, inputdata, outputdata):
        self.inputdata = inputdata
        self.answer = outputdata
        for data in self.inputdata:
            self.decision.append(self.brain.feedForward(data))

    def crossover(self, parent2):
        child = Player()
        child.brain = self.brain.crossover(parent2.brain)
        child.brain.generateNetwork()
        return child

    def clone(self):
        return self
