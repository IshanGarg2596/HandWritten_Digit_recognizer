from sklearn.datasets import fetch_mldata
import numpy as np
custom_data_home = '~/Desktop/Handwritten digit recognition'
mnist = fetch_mldata('MNIST original', data_home=custom_data_home)
print(mnist.data.shape)
print(np.unique(mnist.target))
