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
                out = np.squeeze(np.asarray(layer))
                print("-Result-")
                print(out)
                print("\n")
                return out

        except (ValueError, IndexError) as err:
            print("\n---------")
            print( "Error: " + str(err) )
            print( "Layer index: " + str(layerIndex) )
            print( "Max layer index: " + str(maxLayer) )

    def gradient( dCost:float, prop:list ):
        propLen = len(prop)
        gradient = [None] * propLen
        for i in range( propLen, 0, -1 ):
            if( i == propLen ):
                gradient[i] = dCost / prop[i]
            else:
                gradient[i] = dCost / (prop[i] + gradient[i+1])

        return gradient

    def learn( inp:np.array, weights:list, bias:list, theta:float ):
        # Calculate the derivative for:
        # Cost in respect to weights
        # Cost in respect to biases

        res1 = AIlib.think( inp, weights, bias ) # Think the first result
        cost1 = AIlib.calcCost( inp, res1 ) # Calculate the cost of the thought result

        inp2 = np.asarray( inp + theta ) # make the new input with `theta` as diff
        res2 = AIlib.think( inp2, weights, bias ) # Think the second result
        cost2 = AIlib.calcCost( inp2, res2 ) # Calculate the cost

        dCost = cost2 - cost1 # get the difference

        weightDer = AIlib.gradient( dCost, weights )
        biasDer = AIlib.gradient( dCost, bias )

        print(weights, len(weights))
