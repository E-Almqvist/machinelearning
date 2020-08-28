#!/usr/bin/env python

from lib.func import AIlib as ai

class rgb(object):
    def __init__(self, loadedWeights = None, loadedBias = None):

        if( not loadedWeights or not loadedBias ):
            self.weights = [ ai.genRandomMatrix(3, 4), ai.genRandomMatrix(4, 4), ai.genRandomMatrix(4, 3) ] # array of matrices of weights
            # 3 input neurons -> 4 hidden neurons -> 4 hidden neurons -> 3 output neurons

            # Will be needing biases too
            self.bias = [ ai.genRandomMatrix(1, 4), ai.genRandomMatrix(1, 4), ai.genRandomMatrix(1, 3) ]
            # This doesn't look very good, but it works so...
            # This is all we need 
        else: # if we want to load our progress from before then this would do it
            print("Loading neural net...")
            self.weights = loadedWeights
            self.bias = loadedBias

    def think(self, inputMatrix):
        print(self.weights)
        print(self.bias)
def init(): # init func
    bot = rgb()

    bot.think(1)

init()
