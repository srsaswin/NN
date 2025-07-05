'''
doing multiple things in parallel s we use batches 
and alse for help with generalization->batch size if the size of point the 
we tell the singlr neuron to fit it 

here we going to calaculate different input for a same layer of nodes
'''

import numpy as np

inputs = [[1,2,3,2.5],   #this are a input at different point on time 
          [2.0,5.0,-1.0,2.0],     #by doing thing this we can process ti in parallel    
          [-1.5,2.7,3.3,-0.8]]

weights1 = [[0.2,0.8,-0.5,1.0],
            [0.5,-0.91,0.26,-0.5],
            [-0.26,-0.27,0.17,0.87]]

biases1  = [2,3,0.5]

'''
np.array(inputs) we us this to converte the list of list to numpy array object
and it have .T for transpose to mak (3 x 4).T -> (4 x 3)
then we multiple it by (3 x 4) * (4 x 3) -> (3 x 3)  
'''

outputs1 = np.dot(inputs,np.array(weights1).T) + biases1

'''
for second layer newron get the first layer outputs as input
'''

weights2 = [[0.1,-0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]

biases2 = [-1,2,-0.5]

outputs2 = np.dot(outputs1,np.array(weights2).T) + biases2

print(outputs1,"\n\n",outputs2)