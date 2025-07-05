'''
doing multiple things in parallel s we use batches 
and alse for help with generalization->batch size if the size of point the 
we tell the singlr neuron to fit it 

here we going to calaculate different input for a same layer of nodes
'''

import numpy as np

inputs = [[1,2,3,2.5],   #this are a input at different point on time 
         [2,5,-1,2],     #by doing thing this we can process ti in parallel    
         [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5,1.0],
          [0.5,-0.91,0.26,-0.5],
          [-0.26,-0.27,0.17,0.87]]

biases = [2,3,0.5]

output = np.dot(weights,np.array(inputs).T) + biases

'''
np.array(inputs) we us this to converte the list of list to numpy array object
and it have .T for transpose to mak (3 x 4).T -> (4 x 3)
then we multiple it by (3 x 4) * (4 x 3) -> (3 x 3)  
'''

print(output)