import tensorflow as tf
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
W = tf.random_normal(shape=[784, 500])
x = tf.random_normal(shape=[100, 784])

a = tf.matmul(x, W)

u = tf.random_uniform(shape=[784])
ind = tf.reshape(tf.where(tf.greater(u, 0.8)), [-1])

b = tf.matmul(tf.gather(x, ind, axis=1), tf.gather(W, ind, axis=0))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
dummy = tf.identity(tf.constant(3.0))
for i in range(10000):
    sess.run([a,b])

st = time.time()
sess.run(a)
print time.time() - st

st = time.time()
sess.run(b)
print time.time() - st
