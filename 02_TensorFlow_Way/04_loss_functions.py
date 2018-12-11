#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 04_loss_functions.py 
@time: 2018/12/06 
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

x_vals = tf.linspace(-1.,1.,500)

target = tf.constant(0.)

# l2 loss 是一种罪常见的回归损失函数 L = (pred - actual)^2

l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

# l1 loss L = abs(pred - actual)

l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

# Pseudo-Huber Loss 在变化值比较大的时候，是L1 Loss的比较平滑近似，当变化值比较小的时候，比较近似于L2 Loss
# L = delta^2*(sqrt(1+((pred-actual)/delta)^2)-1)

# Pseudo-Huber with delta = 0.25
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1),tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

# Pseudo-Huber with delta = 5
delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2),tf.sqrt(1. + tf.square((target - x_vals)/delta1)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label = 'L2 Loss')
plt.plot(x_array,l1_y_out, 'r--', label = 'L1 Loss')
plt.plot(x_array, phuber1_y_out, 'k-.', label = 'P-Huber Loss(0.25)')
plt.plot(x_array, phuber2_y_out, 'g:', label = 'P-Huber Loss(0.5)')
plt.ylim(-0.2, 0.4)
plt.legend(loc = 'lower right', prop = {'size': 11})
plt.grid()
plt.show()
