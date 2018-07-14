import tensorflow as tf
import numpy as np
#
# a = tf.Variable([1,2,3,4])
# a1 = [1,1,3,20,2]
# a1 = tf.Variable(a1,dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     p = tf.placeholder(dtype=tf.int32, shape=None)
#     b = p + 1
#     print(a.eval())
#     print(b.eval({p:[1,2,3,4]}))
#     print(tf.reduce_mean(a1).eval())

# a = tf.random_normal([2,4],seed=2)
# tf.get_variable()
# tf.Variable()
# with tf.Session() as sess:
#     print(sess.run(a))

# def foo():
#   with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
#     v = tf.get_variable("v", [1])
#   return v
#
# v1 = foo()  # Creates v.
# # print(v1.eval())
# with tf.Session() as sess:
#     sess.run(v1)
# v2 = foo()  # Gets the same, existing v.
# print(v1 == v2)

# a = tf.random_uniform([1])
# b = tf.random_normal([1])
#
# # Repeatedly running this block with the same graph will generate the same
# # sequence of values for 'a', but different sequences of values for 'b'.
# print("Session 1")
# with tf.Session() as sess1:
#   print(sess1.run(a))  # generates 'A1'
#   print(sess1.run(a))  # generates 'A2'
#   print(sess1.run(a))
#   print(sess1.run(b))  # generates 'B1'
#   print(sess1.run(b))  # generates 'B2'
#
# print("Session 2")
# with tf.Session() as sess2:
#   print(sess2.run(a))  # generates 'A1'
#   print(sess2.run(a))  # generates 'A2'
#   print(sess2.run(b))  # generates 'B3'
#   print(sess2.run(b))  # generates 'B4'

# a = tf.Variable([1,2,3,4,5,5,6])
# b = tf.Variable([1,2,3,4,5,6,7])
# c = tf.square(b)
from docutils.nodes import label

a = tf.Variable([[0,0,0,1,0,0],[0,0,0,1,0,0]])
b = tf.Variable([[0.1,0.1,0,0.5,0.2,0.1],[0.1,0.1,0,0.5,0.2,0.1]])



with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())
    c = tf.nn.softmax_cross_entropy_with_logits(labels=a,logits=b)
    print(c.eval())


# a = tf.constant([1,2,3,4])
# # a = [[1,2,3,4],[1,2,3,5],[2,3,4,5]]
# b = [0,1,1,1,1]
#
# with tf.Session() as sess :
#     sess.run(tf.global_variables_initializer())
#     c = tf.expand_dims(a,1)
#     print(c.eval())

# a = np.array([[1,2,3]])
# print(a)
# print(a.shape)
# b = a.reshape([3])
# print(b)
# print(b.shape)