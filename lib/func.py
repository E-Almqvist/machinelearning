import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def correctFunc(inp:np.array): # generates the correct answer for the AI 
        return np.array( rgb[2], rgb[1], rgb[0] ) # basically invert the rgb values

    def calcCost( inp:np.array, out:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
        sumC = 0
        outLen = len(out)

        correctOut = correctFunc(inp) # the "correct" output

        for i in range(outLen):
            sumC += (out[i] - correctOut[i])**2 # get the difference of every value

        return sumC / outLen # return the average cost of all rows

    def genRandomMatrix( x:int, y:int, min: float=0.0, max: float=1.0 ): # generate a matrix with x, y dimensions with random values from min-max in it
        return np.random.rand(x, y)

    def think( inp:np.matrix, weights:list, bias:list, layerIndex: int=0 ): # recursive thinking, hehe
        # the length of weights and bias should be the same
        # if not then the neural net is flawed/incorrect
        maxLayer = len(weights)
        biasLen = len(bias)
        if( maxLayer != len(bias) ):
            print("Neural Network Error: Length of weights and bias are not equal.")
            print("Weights: ${maxLayer}, Bias: ${biasLen}")
            exit()

        weightedInput = np.dot( weights[layerIndex], inp ) # dot multiply the input and the weights
        layer = np.add( weightedInput, bias[layerIndex] ) # add the biases

        if( layerIndex >= maxLayer ):
            return layer
        else:
            think( layer, weights, bias, layerIndex + 1 )
