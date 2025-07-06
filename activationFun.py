import numpy as np

'''
for study e use nnfs 
nnfs.datasets give a common dataset to check the output with youtube vedio
nnfs.init() it overwrite the numpy to set is a common value (at default it can set in int or float) 
and also over write numpy.random.seed() to get same output
'''

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

X,y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weight = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,input):
        self.output = np.dot(input,self.weight) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)