import numpy as np
from copy import deepcopy as copy

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def correctFunc(inp:np.array): # generates the correct answer for the AI 
	return np.asarray( [1.0 - inp[0], 1.0 - inp[1], 1.0 - inp[2]] ) # basically invert the rgb values

def calcCost( predicted:np.array, correct:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
	costSum = 0
	maxLen = len(correct)

	for i in range(maxLen):
		costSum += abs((predicted[i] - correct[i]))

	return costSum / maxLen

def getThinkCost( inp:np.array, predicted:np.array ):
	corr = correctFunc(inp)
	return calcCost( predicted, corr )

def genRandomMatrix( x:int, y:int, min: float=0.0, max: float=1.0 ): # generate a matrix with x, y dimensions with random values from min-max in it
	# apply ranger with * and -
	mat = np.random.rand(x, y) - 0.25
	return mat

def think( inp:np.array, obj, layerIndex: int=0 ): # recursive thinking, hehe
	maxLayer = len(obj.weights) - 1
	weightedLayer = np.dot( inp, obj.weights[layerIndex] ) # dot multiply the input and the weights
	layer = sigmoid( np.add(weightedLayer, obj.bias[layerIndex]) ) # add the biases

	if( layerIndex < maxLayer ):
		return think( layer, obj, layerIndex + 1 )
	else:
		out = np.squeeze(np.asarray(layer))
		return out

def propDer( dCost, dProp ):
	# Calculate the partial derivative for that prop
	return dCost / dProp

def compareAIobjects( inp, obj1, obj2 ):
	# Compare the two instances
	res1 = think( inp, obj1 )
	cost1 = getThinkCost( inp, res1 ) # get the cost

	res2 = think( inp, obj2 )
	cost2 = getThinkCost( inp, res2 ) # get the second cost

	# Actually calculate stuff 
	dCost = cost2 - cost1
	return dCost, cost1

def compareInstanceWeight( obj, inp, theta:float, layerIndex:int, neuronIndex_X:int, neuronIndex_Y:int ):
	# Create new a instance of the object
	obj2 = copy(obj) # annoying way to create a new instance of the object

	obj2.weights[layerIndex][neuronIndex_X][neuronIndex_Y] += theta # mutate the second objects neuron
	dCost, curCost = compareAIobjects( inp, obj, obj2 ) # compare the two and get the dCost with respect to the weights

	return dCost, curCost

def compareInstanceBias( obj, inp, theta:float, layerIndex:int, biasIndex:int ):
	obj2 = copy(obj)

	obj2.bias[layerIndex][0][biasIndex] += theta # do the same thing for the bias
	dCost, curCost = compareAIobjects( inp, obj, obj2 )

	return dCost, curCost

def getChangeInCost( obj, inp, theta, layerIndex ):
	mirrorObj = copy(obj)

	# Fill the buffer with None so that the dCost can replace it later
	dCost_W = np.zeros( shape = mirrorObj.weights[layerIndex].shape ) # fill it with a placeholder
	dCost_B = np.zeros( shape = mirrorObj.bias[layerIndex].shape )

	# Get the cost change for the weights
	weightLenX = len(dCost_W)
	weightLenY = len(dCost_W[0])

	for x in range(weightLenX): # get the dCost for each x,y
		for y in range(weightLenY):
			dCost_W[x][y], curCostWeight = compareInstanceWeight( obj, inp, theta, layerIndex, x, y )

	# Get the cost change for the biases
	biasLenY = len(dCost_B[0])
	for index in range(biasLenY):
		dCost_B[0][index], curCostBias = compareInstanceBias( obj, inp, theta, layerIndex, index )

	return dCost_W, dCost_B, (curCostBias + curCostWeight)/2



def gradient( inp:np.array, obj, theta:float, maxLayer:int, layerIndex: int=0, grads=None, obj1=None, obj2=None ): # Calculate the gradient for that prop
	# Check if grads exists, if not create the buffer
	if( not grads ):
		grads = [None] * (maxLayer+1)
		
	dCost_W, dCost_B, meanCurCost = getChangeInCost( obj, inp, theta, layerIndex )

	# Calculate the gradient for the layer
	weightDer = propDer( dCost_W, theta )
	biasDer = propDer( dCost_B, theta )

	# Append the gradients to the list
	grads[layerIndex] = {
		"weight": weightDer,
		"bias": biasDer
	}

	newLayer = layerIndex + 1
	if( newLayer <= maxLayer ):
		return gradient( inp, obj, theta, maxLayer, newLayer, grads, obj1, obj2 )
	else:
		return grads, dCost_W, dCost_B, meanCurCost
	
def calculateSteepness( cost:float, gradient:np.matrix ):
	gradLen = np.linalg.norm( gradient ) # basically calculate the hessian but transform the gradient into a scalar (its length)
	ddCost = cost / gradLen
	out = np.log10(ddCost)

	return out 

def getLearningRate( cost:float, gradient:np.matrix, maxLen:int ):
	learningrate = {
		"weight": [None] * maxLen,
		"bias": [None] * maxLen
	}

	for i in range(maxLen):
		learningrate["weight"][i] = calculateSteepness( cost, gradient ) 
		learningrate["bias"][i] = calculateSteepness( cost, gradient )

	return learningrate


def mutateProps( inpObj, curCost:float, maxLayer:int, gradient:list ):
	obj = copy(inpObj)

	for layer in range(maxLayer):
		lr = getLearningRate( curCost, gradient[layer]["weight"], maxLayer )
		print(lr)

		obj.weights[layer] -= lr["weight"] * gradient[layer]["weight"] # mutate the weights
		obj.bias[layer] -= lr["bias"] * gradient[layer]["bias"]
		# obj.weights[i] -= obj.learningrate * gradient[i]["weight"] # mutate the weights
		# obj.bias[i] -= obj.learningrate * gradient[i]["bias"]


	return obj

def learn( inputNum:int, targetCost:float, obj, theta:float, curCost: float=None ):
	# Calculate the derivative for:
	# Cost in respect to weights
	# Cost in respect to biases

	# i.e. : W' = W - lr * gradient (respect to W in layer i) = W - lr*[ dC / dW[i] ... ]
	# So if we change all the weights with i.e. 0.01 = theta, then we can derive the gradient with math and stuff

	
	while( not curCost or curCost > targetCost ): # targetCost is the target for the cost function
		inp = np.asarray(np.random.rand( 1, inputNum ))[0] # create a random learning sample

		maxLen = len(obj.bias)
		grads, costW, costB, curCost = gradient( inp, obj, theta, maxLen - 1 )

		obj = mutateProps( obj, curCost, maxLen, grads ) # mutate the props for next round
		print(f"Cost: {curCost}")


	print("DONE\n")
	print(obj.weights)
	print(obj.bias)
