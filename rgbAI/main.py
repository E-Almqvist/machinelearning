#!/usr/bin/env python
import numpy as np
from lib.ailib import ai

class rgb(object):
	def __init__(self, loadedWeights: np.matrix=None, loadedBias: np.matrix=None):

		if( not loadedWeights or not loadedBias ): # if one is null (None) then just generate new ones
			print("Generating weights and biases...")
			self.weights = [ ai.genRandomMatrix(3, 3), ai.genRandomMatrix(3, 3) ] # array of matrices of weights
			# 3 input neurons -> 3 hidden neurons -> 3 hidden neurons -> 3 output neurons

			# Generate the biases
			self.bias = [ ai.genRandomMatrix(1, 3), ai.genRandomMatrix(1, 3) ]
			# This doesn't look very good, but it works so...

			print( self.weights )
			print( self.bias )

		else: # if we want to load our progress from before then this would do it
			self.weights = loadedWeights
			self.bias = loadedBias

	def calcError( self, inp:np.array, out:np.array ):
		cost = ai.calcCost( inp, out )
		# Cost needs to get to 0, we can figure out this with backpropagation
		return cost

	def learn( self ):
		ai.learn( 3, 0.001, self, 0.001 )

	def think( self, inp:np.array ):
		print("\n-Input-")
		print(inp)

		res = ai.think( inp, self )

		print("\n-Output-")
		print(res)
		return res

def init():
	bot = rgb()
	bot.learn()

	#inpArr = np.asarray([1.0, 1.0, 1.0])
	#res = bot.think( inpArr )
	#err = bot.calcError( inpArr, res )
	#print(err)

init()
