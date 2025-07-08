import nnfs
import numpy as np

from nnfs.datasets import spiral_data

nnfs.init()

''''
we connot use ReUL in output layer so we use this 
so we use softmax
'''



'''
conside we have two neuoran to pridect dog or cat

the first problam is we need the output shoud be in 0 or 1
but we are not going to get it we get 0.45 and 0.55

it means it is 0.45% cat and 0.55% dog if the correct answer is dog then we have 0.45% error in it
and the next problam is negative values we cont gest remove it or conver to 0 using ReLU
becouse the negative value is alse a value ewhich have a part inthe pridection and in error correction

so we use softmax
'''

'''
first step is to rize to e pow 

for ex :
layer_outputs = [4.8, -1.21, 2.385]

output: [121.51041752   3.35348465  10.85906266]

second step is to place this all in 0 to 1 

conside we have three neuorn then the output is

N_t = N_1 + N_2 + N_3

output = [N_1 / N_t , N_2 / N_t , N_3 / N_t]

output:[0.8952826639572619,0.02470830678209937,0.0800090292606387]
'''

'''
there is one problan 

pow(e,-1) : ok
pow(e,100) : ok
pow(e,1000) : error overflow

so we used to sub the max ot that layer to ever other input before exp operation
but there is no change in final output
'''


layer_outputs = [4.8, 1.21, 2.385]

exp_value = np.exp(layer_outputs)
norm_values = exp_value / np.sum(exp_value)

# print(norm_values)

''''
for batch of inputs
'''

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)
'''
np.sum(exp_values,axis=1,keepdims=True)
here it is used to get proper shape
'''

norm_values = exp_values / np.sum(exp_values,axis=1,keepdims=True)

# print(norm_values)

'''
as is say the is over flow problam with exp()
so we use:  
    v = u - max u
    u : inputs
    v : each element in inputs - by max in that perticular inputs

then the exp is range in -inf to 0
'''


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.outputs = np.maximum(0,inputs)

class Activation_softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        self.outputs = exp_values / np.sum(exp_values,axis=1,keepdims=True)

X,y = spiral_data(samples=100,classes=3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3,3)
activation2 = Activation_softmax()

dense1.forward(X)
activation1.forward(dense1.outputs)

dense2.forward(activation1.outputs)
activation2.forward(dense2.outputs)

print(activation2.outputs[:5])