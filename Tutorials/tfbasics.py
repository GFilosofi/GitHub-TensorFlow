# Python
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3])
x = tf.placeholder(tf.float32)
y = W * x + b
real_y = tf.placeholder(tf.float32)
delta_y = tf.square(y - real_y)
cost = tf.reduce_sum(delta_y)

#fixW = tf.assign(W, [-3.])
#fixb = tf.assign(b, [3.])

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cost)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
#	sess.run([fixW, fixb])
	for i in range(1000):
		sess.run(train, {x: [1, 2, 3, 4], real_y: [0, -3, -6, -9]})
	print(sess.run([W,b]))
	print('cost:', sess.run(cost, {x: [1, 2, 3, 4], real_y: [0, -3, -6, -9]}))
