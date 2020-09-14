import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def sigmoid_der(x):
        return AIlib.sigmoid(x) * (1 - AIlib.sigmoid(x))

    def correctFunc(inp:np.array): # generates the correct answer for the AI 
        return np.array( [inp[2], inp[1], inp[0]] ) # basically invert the rgb values

    def calcCost( predicted:np.array, correct:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
        return (predicted - correct)**2

    def calcCost_derv( predicted:np.array, correct:np.array ):
        return (predicted - correct)*2

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

    def gradient( cost:float, inp:np.array, predicted:np.array, correct:np.array ):
        # Calculate the gradient
        derv1 = AIlib.calcCost_derv( predicted, correct )
        derv2 = AIlib.sigmoid_der( predicted )

        gradient = np.transpose( np.asmatrix(derv1 * derv2 * inp) )
        print("Inp:", inp)
        print("Grad:", gradient)
        return gradient


    def mutateProp( prop:list, lr, gradient ):
        newProp = [None] * len(prop)

        for i in range(len(prop)):
            newProp[i] = prop[i] - (lr*gradient)

        return newProp

    def learn( inp:np.array, obj, theta:float ):
        # Calculate the derivative for:
        # Cost in respect to weights
        # Cost in respect to biases

        predicted = AIlib.think( inp, obj.weights, obj.bias ) # Think the first result
        correct = AIlib.correctFunc( inp )
        cost = AIlib.calcCost( predicted, correct ) # Calculate the cost of the thought result

        #inp2 = np.asarray( inp + theta ) # make the new input with `theta` as diff
        #res2 = AIlib.think( inp2, obj.weights, obj.bias ) # Think the second result
        #cost2 = AIlib.calcCost( inp2, res2 ) # Calculate the cost

        gradient = AIlib.gradient( cost, inp, predicted, correct )

        obj.weights = AIlib.mutateProp( obj.weights, obj.learningrate, gradient )
        obj.bias = AIlib.mutateProp( obj.bias, obj.learningrate, gradient )

        print("Cost: ", cost1)
