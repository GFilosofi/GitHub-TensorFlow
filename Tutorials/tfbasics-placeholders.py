# Python
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
result = tf.multiply(a,b)
#print(a,b,result)

sess = tf.Session()
output = sess.run(result, feed_dict={a: .1, b: [5.0, 2.0]})
print(output)
