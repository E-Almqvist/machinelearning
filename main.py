#!/usr/bin/env python

from lib.func import *

class rgb(object):
    def __init__(self):
        self.weights = 1

    def think(self, inputMatrix):
        print("thonk: " + str( sigmoid(self.weights) ))

def init(): # init func
    bot = rgb()

    bot.think(1)

init()
