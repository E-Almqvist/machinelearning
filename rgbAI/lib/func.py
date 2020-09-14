import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(x):
        return AIlib.sigmoid(x) * (1 - AIlib.sigmoid(x))

    def correctFunc(inp:np.array): # generates the correct answer for the AI 
        return np.array( [inp[2], inp[1], inp[0]] ) # basically invert the rgb values

    def calcCost( inp:np.array, out:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
        sumC = 0
        outLen = len(out)

        correctOut = AIlib.correctFunc(inp) # the "correct" output

        diff = (out - outLen)**2
        sumC = diff.sum()

        return sumC / outLen # return the cost

    def genRandomMatrix( x:int, y:int, min: float=0.0, max: float=1.0 ): # generate a matrix with x, y dimensions with random values from min-max in it
        # apply ranger with * and -
        mat = np.random.rand(x, y) - 0.25
        return mat

    def think( inp:np.array, weights:list, bias:list, layerIndex: int=0 ): # recursive thinking, hehe
        maxLayer = len(weights) - 1
        weightedInput = np.dot( inp, weights[layerIndex] ) # dot multiply the input and the weights
        layer = AIlib.sigmoid( np.add(weightedInput, bias[layerIndex]) ) # add the biases

        if( layerIndex < maxLayer ):
            return AIlib.think( layer, weights, bias, layerIndex + 1 )
        else:
            out = np.squeeze(np.asarray(layer))
            print("-Result-")
            print(out)
            print("\n")
            return out

    def gradient( dCost:float, out:np.array, inp:np.array ):
        # Calculate the gradient
        print("")

    def mutateProp( prop:list, gradient:list ):
        newProp = [None] * len(gradient)

        for i in range(len(gradient)):
            newProp[i] = prop[i] - gradient[i] # * theta (relative to slope or something)

        return newProp

    def learn( inp:np.array, obj, theta:float ):
        # Calculate the derivative for:
        # Cost in respect to weights
        # Cost in respect to biases

        res1 = AIlib.think( inp, obj.weights, obj.bias ) # Think the first result
        cost1 = AIlib.calcCost( inp, res1 ) # Calculate the cost of the thought result

        #inp2 = np.asarray( inp + theta ) # make the new input with `theta` as diff
        #res2 = AIlib.think( inp2, obj.weights, obj.bias ) # Think the second result
        #cost2 = AIlib.calcCost( inp2, res2 ) # Calculate the cost

        dCost = cost1 # get the difference # cost2 - cost1

        weightDer = AIlib.gradient( dCost, theta, obj.weights )
        biasDer = AIlib.gradient( dCost, theta, obj.bias )

        obj.weights = AIlib.mutateProp( obj.weights, weightDer )
        obj.bias = AIlib.mutateProp( obj.bias, biasDer )

        print("Cost: ", cost1)
