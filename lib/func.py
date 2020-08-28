import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def correctFunc(rgb): # generates the correct answer for the AI 
    return ( rgb[2], rgb[1], rgb[0] ) # basically invert the rgb values

def genRandomMatrix( x:int, y:int ):
    return np.random.rand(x, y)
