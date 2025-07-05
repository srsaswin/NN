'''
to find the output for a four nodes we use dot product 
if give output from a one whole layer
the current layer have 3 nodes and pre layer have 4 nodes
'''

inputs = [1,2,3,2.5] #input form 4 different node from pre layer
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]] #weigth of each node to current node

biases = [2,3,0.5]

without_numpy = []
for node_bias,node_weights in zip(biases,weights):
    node_output = 0
    for individual_input,individual_weight in zip(inputs,node_weights):
        node_output += individual_input * individual_weight
    node_output += node_bias
    without_numpy.append(node_output)

'''
now the implementation using numpy
'''

import numpy as np

with_numpy = np.dot(weights,inputs) + biases 

'''
the resion we pass the weights as a first parameter is to tell i need a output that 
need to be in length for more detels you can see the func of dot
if order changs it give shape error
'''

print(without_numpy,with_numpy)
