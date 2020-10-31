# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 19:06
# @Author  : HH
# @File    : Species.py
# @Project : ConvTest
from FuncProcess import *
from typing import *
from Genome import *
import numpy as np


class Species:

    def __init__(self, p=None):
        self.players = []  # type:list[Player]
        self.bestfitness = 0
        self.champ = None  # type:Player
        self.averagefitness = 0
        self.staleness = 0
        self.rep = None  # type:Genome
        self.excessCoeff = 1.5
        self.weightDiffCoeff = 0.8
        self.compatibilityThreshold = 1
        if p is not None:
            self.players.append(p)
            self.bestfitness = p.fitness
            self.rep = p.brain.clone()
            self.champ = p.clone()

    def samespecies(self, g) -> bool:  # 种群划分
        excessAndDisjoint = self.getExcessDisjoint(g, self.rep)
        averageWeightDiff = self.averageWeightDiff(g, self.rep)
        largeGenomeNormaliser = 1
        compatibility = (self.excessCoeff * excessAndDisjoint / largeGenomeNormaliser) + (
                self.weightDiffCoeff * averageWeightDiff)
        return self.compatibilityThreshold > compatibility

    def addToSpecies(self, p):
        self.players.append(p)

    def getExcessDisjoint(self, brain1, brain2) -> float:
        matching = 0.0
        for i in range(len(brain1.genes)):
            for j in range(len(brain2.genes)):
                if brain1.genes[i].innovationNo == brain2.genes[j].innovationNo:
                    matching += 1
                    break
        return len(brain1.genes) + len(brain2.genes) - 2 * matching

    def averageWeightDiff(self, brain1, brain2) -> float:
        matching = 0
        totaldiff = 0
        for i in range(len(brain1.genes)):
            for j in range(len(brain2.genes)):
                if brain1.genes[i].innovationNo == brain2.genes[j].innovationNo:
                    matching += 1
                    totaldiff += abs(brain1.genes[i].weight - brain2.genes[j].weight)
                    break
        if matching == 0:
            return 100

        return totaldiff / matching

    def sortSpecies(self):
        for i in range(len(self.players)):
            for j in range(len(self.players) - 1):
                if self.players[j].fitness < self.players[j + 1].fitness:
                    tempplayer = self.players[j + 1]
                    self.players[j + 1] = self.players[j]
                    self.players[j] = tempplayer

        # for i in range(len(self.players)):
        #     max = 0
        #     maxindex = 0
        #     for j in range(len(self.players)):
        #         if self.players[j].fitness > max:
        #             max = self.players[j].fitness
        #             maxindex = j
        #     temp.append(self.players[maxindex])
        #     self.players.remove(maxindex)
        #     i -= 1
        if len(self.players) == 0:
            self.staleness = 16
            return
        if self.players[0].fitness > self.bestfitness:
            self.staleness = 0
            self.bestfitness = self.players[0].fitness
            self.rep = self.players[0].brain.clone()
            self.champ = self.players[0].clone()
        else:
            self.staleness += 1

    def setaverage(self):
        sumcount = 0
        for i in range(len(self.players)):
            sumcount += self.players[i].fitness
        if len(self.players) == 0:
            self.averagefitness = sumcount / 1
        else:
            self.averagefitness = sumcount / len(self.players)

    def playerBrith(self, innovationHistory):
        baby = None  # type: Player
        if np.random.uniform(0, 1) < 0.25:
            baby = self.selectplayer().clone()
        else:

            parent1 = self.selectplayer()
            parent2 = self.selectplayer()
            if parent1.fitness < parent2.fitness:
                baby = parent2.crossover(parent1)
            else:
                baby = parent1.crossover(parent2)
        baby.brain.mutate(innovationHistory)
        return baby

    def selectplayer(self):
        fitnesssum = 0
        for i in range(len(self.players)):
            fitnesssum += self.players[i].fitness
        rand = np.random.uniform(0, fitnesssum)
        runningsum = 0
        for i in range(len(self.players)):
            runningsum += self.players[i].fitness
            if runningsum > rand:
                return self.players[i]
        return self.players[0]

    def fitnesssharing(self):
        for i in range(len(self.players)):
            self.players[i].fitness /= len(self.players)

    def cull(self):
        if len(self.players) > 3:
            dellist = []
            for i in range(math.ceil(len(self.players) / 2), len(self.players) - 1):
                dellist.append(self.players[i])
            for i in dellist:
                self.players.remove(i)
