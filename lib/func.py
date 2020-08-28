import numpy as np

class AIlib:
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def correctFunc(rgb): # generates the correct answer for the AI 
        return ( rgb[2], rgb[1], rgb[0] ) # basically invert the rgb values

    def genRandomMatrix( x:int, y:int ): # generate a matrix with x, y dimensions with random values from 0-1 in it
        return np.random.rand(x, y)
