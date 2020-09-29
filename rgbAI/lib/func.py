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

    def think( inp:np.array, obj, layerIndex: int=0 ): # recursive thinking, hehe
        maxLayer = len(obj.weights) - 1
        weightedLayer = np.dot( inp, obj.weights[layerIndex] ) # dot multiply the input and the weights
        layer = AIlib.sigmoid( np.add(weightedLayer, obj.bias[layerIndex]) ) # add the biases

        if( layerIndex < maxLayer ):
            return AIlib.think( layer, obj, layerIndex + 1 )
        else:
            out = np.squeeze(np.asarray(layer))
            return out

    def propDer( dCost, dProp ):
        # Calculate the partial derivative for that prop
        return dCost / dProp

    def gradient( inp:np.array, obj, prop, theta ):
        # Calculate the gradient for that prop
        prop2 = prop + theta
        # then create another instance of the object and compare

        # calculate the diff between the new prop and old
        res = AIlib.think( inp, obj. )

    def mutateProp( prop:list, lr:float, gradient ):
        newProp = [None] * len(prop)

        for i in range(len(prop)):
            newProp[i] = prop[i] - (lr*gradient)

        return newProp

    def learn( inp:np.array, obj, theta:float ):
        # Calculate the derivative for:
        # Cost in respect to weights
        # Cost in respect to biases

        # i.e. : W' = W - lr * gradient (respect to W in layer i) = W - lr*[ dC / dW[i] ... ]
        # So if we change all the weights with i.e. 0.01 = theta, then we can derive the gradient with math and stuff

