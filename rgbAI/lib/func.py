import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def correctFunc(inp:np.array): # generates the correct answer for the AI 
        return np.array( [inp[2], inp[1], inp[0]] ) # basically invert the rgb values

    def calcCost( predicted:np.array, correct:np.array ): # cost function, lower -> good, higher -> bad, bad bot, bad
        return (predicted - correct)**2

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
        layer = AIlib.sigmoid( np.add(weightedLayer, obj.bias[layerIndex]) ) # add the biases

        if( layerIndex < maxLayer ):
            return AIlib.think( layer, obj, layerIndex + 1 )
        else:
            out = np.squeeze(np.asarray(layer))
            return out

    def propDer( dCost, dProp ):
        # Calculate the partial derivative for that prop
        return dCost / dProp

    def gradient( inp:np.array, obj, theta:float, maxLayer:int, layerIndex: int=0, grads: list=[], obj1=None, obj2=None ):
        # Calculate the gradient for that prop

        # Create new instances of the object
        if( !obj1 or !obj2 ):
            obj1 = obj
            obj2 = obj

        obj2.weights[layerIndex] += theta # mutate the second object
        obj2.bias[layerIndex] += theta

        # Compare the two instances
        res1 = AIlib.think( inp, obj1 )
        cost1 = AIlib.getThinkCost( inp, res1 ) # get the cost

        res2 = AIlib.think( inp, obj2 )
        cost2 = AIlib.getThinkCost( inp, res2 ) # get the second cost

        # Actually calculate stuff 
        dCost = cost2 - cost1
        dWeight = obj2.weights[layerIndex] - obj1.weights[layerIndex]
        dBias = obj2.bias[layerIndex] - obj1.bias[layerIndex]

        # Calculate the gradient for the layer
        weightDer = AIlib.propDer( dCost, dWeight )
        biasDer = AIlib.propDer( dCost, dBias )

        # Append the gradients to the list
        grads[layerIndex] = {
            "weight": weightDer,
            "bias": biasDer
        }

        newLayer = layerIndex + 1
        if( newLayer <= maxLayer ):
            return AIlib.gradient( inp, obj, theta, maxLayer, newLayer, grads, obj1, obj2 )
        else:
            return grads

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

