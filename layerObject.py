import numpy as np

np.random.seed(10)

''''
conside X is a inputs of batch size of 3
'''
X = [[1,2,3,2.5],    
    [2.0,5.0,-1.0,2.0],        
    [-1.5,2.7,3.3,-0.8]]


class Layer_Dense:

    '''
    print(np.random.rand(4,5)) need not mul by 0.1 if we use rand() thr random is in range from 0 to 1 correctly
    [[0.07713206 0.00207519 0.06336482 0.07488039 0.0498507 ]
    [0.02247966 0.01980629 0.07605307 0.01691108 0.00883398]
    [0.06853598 0.09533933 0.00039483 0.05121923 0.0812621 ]
    [0.06125261 0.07217553 0.02918761 0.09177741 0.07145758]]

    print(np.zeros((1,4))) here this function only takes object so we pass tuple
    [[0. 0. 0. 0.]]
    '''

    def __init__(self,n_inputs,n_neurons):
        self.weight = 0.10 * np.random.randn(n_inputs,n_neurons) #here we use randn()
        self.biases = np.zeros((1,n_neurons))

    def forward(self,input):
        self.output = np.dot(input,self.weight) + self.biases

'''
the layer design like 
we have 4 input neurons -> 5 neurons -> 2 neurons finaly
'''

layer1 = Layer_Dense(4,5) 
layer2 = Layer_Dense(5,2)

layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)