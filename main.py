#!/usr/bin/env python
import numpy as np
from lib.func import AIlib as ai

class rgb(object):
    def __init__(self, loadedWeights: np.matrix=None, loadedBias: np.matrix=None):

        if( not loadedWeights or not loadedBias ): # if one is null (None) then just generate new ones
            print("Generating weights and biases...")
            self.weights = [ ai.genRandomMatrix(4, 3), ai.genRandomMatrix(4, 4), ai.genRandomMatrix(3, 4) ] # array of matrices of weights
            # 3 input neurons -> 4 hidden neurons -> 4 hidden neurons -> 3 output neurons

            # Generate the biases
            self.bias = [ ai.genRandomMatrix(4, 1), ai.genRandomMatrix(4, 1), ai.genRandomMatrix(3, 1) ]
            # This doesn't look very good, but it works so...

        else: # if we want to load our progress from before then this would do it
            self.weights = loadedWeights
            self.bias = loadedBias

    def learn():
        print("learn")

    def think(self, inp:np.array):
        res = ai.think( inp, self.weights, self.bias )
        print( "Result: " + str(res) )
        # print(self.weights)
        # print(self.bias)

def init(): # init func
    bot = rgb()
    bot.think( np.array([0.2, 0.4, 0.8]) )

init()
