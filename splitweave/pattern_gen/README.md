# System design

Each level has: 
1. Sampler
2. Edit functions
3. Default.


The sampler should take only the root as input. 

The edit function should take the parent (and also occasionally the root as input)

The Edit functions themselves will only update the current node. So they should take only themselves as input. 


