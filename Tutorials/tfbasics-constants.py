# Python
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

scalar = tf.constant(5.0, tf.float32)	# you can omit the type
vector = tf.constant([.1,.2,0.3], dtype=tf.float32)
result = tf.multiply(scalar, vector)	# a math operation is also a node
#print(scalar,vector,result)		# prints only nodes structure

sess = tf.Session()
output = sess.run([scalar, vector, result])		# evaluates the computational graph defined above
print(output)					# prints the actual numbers
