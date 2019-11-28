import numpy as np
import tensorflow as tf

def make_sqrt_12():

    val = np.sqrt(12.0)
    print(val)

    # This is a sick ass comment.

    a = 2.0 + val
    return a


make_sqrt_12()



#Comparing how NP arrays work compared to vanilla python lists.

a = np.array([1,2,3])  #numPy arrays
tensor_1d = np.array([1,2,3])

b = [1,2,3]  #Python lists
ls = [1,2,3]

print(a)   #   [1 2 3]
print(b)   #   [1, 2, 3]

#array comparison
print(a == tensor_1d) #  [ True  True  True]

#list comparison
print(b == ls) #  True

#list and array comparison
print(a == b)  #   [ True  True  True]
