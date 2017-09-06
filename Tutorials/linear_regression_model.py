# Python

# import libraries
import tensorflow as tf
import numpy as np

%matplotlib inline
import pylab

# Create data
x_data = np.random.rand(100).astype(np.float32)
noise = np.random.normal(scale = 0.01, size = len(x_data))
y_data = x_data * 0.1 + 0.3 + noise

# Plot the Data
pylab.Plot(x_data, y_data, '.')

