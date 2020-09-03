import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def correctFunc(inp:np.array): # generates the correct answer for the AI 
        return np.array( [inp[2], inp[1], inp[0]] ) # basically invert the rgb values

    def calcCost( inp:np.array, out:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
        sumC = 0
        outLen = len(out)

        correctOut = AIlib.correctFunc(inp) # the "correct" output

        for i in range(outLen):
            sumC += (out[i] - correctOut[i])**2 # get the difference of every value

        return sumC # return the cost

    def genRandomMatrix( x:int, y:int, min: float=0.0, max: float=1.0 ): # generate a matrix with x, y dimensions with random values from min-max in it
        # apply ranger with * and -
        mat = np.random.rand(x, y) - 0.25
        return mat

    def think( inp:np.array, weights:list, bias:list, layerIndex: int=0 ): # recursive thinking, hehe
        try:
            maxLayer = len(weights) - 1
            weightedInput = np.dot( inp, weights[layerIndex] ) # dot multiply the input and the weights
            layer = AIlib.sigmoid( np.add(weightedInput, bias[layerIndex]) ) # add the biases

            if( layerIndex < maxLayer ):
                print(weights[layerIndex])
                print("\n")
                print("Layer " + str(layerIndex))
                print(layer)
                print("\n")

            if( layerIndex < maxLayer ):
                return AIlib.think( layer, weights, bias, layerIndex + 1 )
            else:
                return np.squeeze(np.asarray(layer))

        except (ValueError, IndexError) as err:
            print("\n---------")
            print( "Error: " + str(err) )
            print( "Layer index: " + str(layerIndex) )
            print( "Max layer index: " + str(maxLayer) )

    def gradient( cost1:float, cost2:float, inp1:np.array, inp2:np.array ):
        dY = np.asarray(cost2 - cost1)
        dX = np.asarray(inp2 - inp1)
        print(dY, dX)
        return dY / dX

