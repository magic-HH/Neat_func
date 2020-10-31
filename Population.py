# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 20:49
# @Author  : HH
# @File    : Population.py
# @Project : ConvTest
from FuncProcess import *
from ConnectionHistory import *
from Species import *


class Populations:
    def __init__(self, size=None):
        self.pop = []  # type: list[Player]
        self.bestplayer = None  # type: Player
        self.bestspecies = 0
        self.bestfitness = 0
        self.gen = 0
        self.innovationHistory = []  # type: list[connectionHistory]
        self.genPlayers = []  # type: list[Player]
        self.species = []  # type: list[Species]
        if size is not None:
            for i in range(size):
                self.pop.append(Player())
                self.pop[i].brain.generateNetwork()
            self.bestplayer = self.pop[0]

    def savebest(self):
        pass
        """
            保存最佳网络
        """

    def checkloop(self, datain, answer):
        for eachplayer in self.pop:
            eachplayer.pridectData(datain, answer)
            eachplayer.calculateFitness()

    def setbestplayer(self):
        tempbest = self.species[0].players[0]
        self.genPlayers.append(tempbest)
        if tempbest.fitness > self.bestplayer.fitness:
            self.bestplayer = tempbest
            self.bestfitness = tempbest.fitness

    def naturalSelection(self):
        self.speciate()
        self.calculateFitness()
        self.sortSpecies()
        self.cullSpecies()
        self.setbestplayer()
        self.killStaleSpecies()
        self.killbadspecies()
        averageSum = self.getAvgFitnessSum()
        children = []  # type: list[Player]
        for s in self.species:
            if len(s.players) == 0:
                continue
            children.append(s.players[0].clone())
            NoOfChildren = math.floor(s.averagefitness / averageSum * len(self.pop)) - 1
            for i in range(NoOfChildren):
                children.append(s.playerBrith(self.innovationHistory))
        while len(children) < len(self.pop):
            children.append(self.species[0].playerBrith(self.innovationHistory))
        self.pop.clear()
        self.pop = children.copy()
        self.gen += 1  # generation add
        for i in range(len(self.pop)):
            self.pop[i].brain.generateNetwork()

    def speciate(self):
        for s in self.species:
            s.players.clear()
        for i in range(len(self.pop)):
            speciesFound = False
            for s in self.species:
                if s.samespecies(self.pop[i].brain):
                    s.addToSpecies(self.pop[i])
                    speciesFound = True
                    break
            if not speciesFound:
                self.species.append(Species(self.pop[i]))

    def calculateFitness(self):
        for i in range(len(self.pop)):
            self.pop[i].calculateFitness()

    def sortSpecies(self):
        for s in self.species:
            s.sortSpecies()
        # temp = []  # type: list[Species]
        # for i in range(len(self.species)):
        #     max = 0
        #     maxindex = 0
        #     for j in range(len(self.species)):
        #         if self.species[i].bestfitness > max:
        #             max = self.species[j].bestfitness
        #             maxindex = j
        #     temp.append(self.species[maxindex])
        #     self.species.remove(maxindex)
        #     i -= 1

        for i in range(len(self.species)):
            for j in range(len(self.species) - 1):
                if self.species[j].bestfitness < self.species[j + 1].bestfitness:
                    tempspecies = self.species[j + 1]
                    self.species[j + 1] = self.species[j]
                    self.species[j] = tempspecies

    def killStaleSpecies(self):
        if len(self.species) < 2:
            return
        dellist = []
        for i in range(len(self.species)):
            if self.species[i].staleness >= 25:
                dellist.append(self.species[i])
        if len(dellist) >= len(self.species) - 2:
            for i in range(len(dellist) - len(self.species) + 2):
                dellist.pop(np.random.randint(0, len(dellist)))
        for i in dellist:
            self.species.remove(i)

    def killbadspecies(self):
        dellist = []
        for s in self.species:
            if len(s.players) == 0:
                dellist.append(s)
        for i in dellist:
            self.species.remove(i)

        if len(self.species) < 3:
            return
        averageSum = self.getAvgFitnessSum()
        dellist = []
        for i in range(len(self.species)):
            if (self.species[i].averagefitness / averageSum * len(self.pop)) < 1:
                dellist.append(self.species[i])

        if len(dellist) >= len(self.species) - 2:
            for i in range(len(dellist) - len(self.species) + 2):
                dellist.pop(np.random.randint(0, len(dellist)))

        for i in dellist:
            self.species.remove(i)

    def getAvgFitnessSum(self):
        averageSum = 0
        for s in self.species:
            averageSum += s.averagefitness

        return averageSum

    def cullSpecies(self):
        for s in self.species:
            if np.random.uniform(-2, 10) < (-0.5 + len(s.players)):
                s.cull()
            s.fitnesssharing()
            s.setaverage()
            if len(s.players) == 0:
                self.species.remove(s)
