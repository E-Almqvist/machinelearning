#!/usr/bin/env python
import numpy as np
from lib.func import AIlib as ai

class rgb(object):
    def __init__(self, loadedWeights: np.matrix=None, loadedBias: np.matrix=None):

        if( not loadedWeights or not loadedBias ): # if one is null (None) then just generate new ones
            print("Generating weights and biases...")
            self.weights = [ ai.genRandomMatrix(3, 4), ai.genRandomMatrix(4, 4), ai.genRandomMatrix(4, 3) ] # array of matrices of weights
            # 3 input neurons -> 4 hidden neurons -> 4 hidden neurons -> 3 output neurons

            # Generate the biases
            self.bias = [ ai.genRandomMatrix(1, 4), ai.genRandomMatrix(1, 4), ai.genRandomMatrix(1, 3) ]
            # This doesn't look very good, but it works so...

            self.generation = 0

            print( self.weights )
            print( self.bias )

        else: # if we want to load our progress from before then this would do it
            self.weights = loadedWeights
            self.bias = loadedBias

    def calcError( self, inp:np.array, out:np.array ):
        cost = ai.calcCost( inp, out )
        # Cost needs to get to 0, we can figure out this with backpropagation
        return cost

    def learn( self, inp:np.array, theta:float ):
        ai.learn( inp, self, theta )

    def think( self, inp:np.array ):
        print("-----Gen " + str(self.generation) + "------")
        print("\n-Input-")
        print(inp)
        print("\n")

        res = ai.think( inp, self.weights, self.bias )

        print("\n-Output-")
        print(res)
        print("\n----------------\n\n")
        return res

    def train( self ):
        for i in range(self.traintimes):
            inpArr = np.asarray(np.random.rand( 1, 3 ))[0]
            print("##################", inpArr)
            self.generation = i
            self.learn( inpArr, 0.1 )

def init(): # init
    bot = rgb()

    bot.traintimes = 100
    bot.train()


init()
