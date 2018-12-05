#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 06_activation_functions.py  激活函数
@time: 2018/12/03 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# 初始化横坐标用于绘图
x_vals = np.linspace(start=-10.,stop=10.,num=100)

# ReLU 激活
print(sess.run(tf.nn.relu([-3.,3.,10.])))
y_relu = sess.run(tf.nn.relu(x_vals))

# ReLU-6 激活
print(sess.run(tf.nn.relu6([-3.,3.,10.])))
y_relu6=sess.run(tf.nn.relu(x_vals))

# Sigmoid 激活
print(sess.run(tf.nn.sigmoid([-1.,0.,1.])))
y_sigmoid = sess.run(tf.nn.sigmoid(x_vals))

# Hyper Tangent 激活
print(sess.run(tf.nn.tanh([-1., 0.,1.])))
y_tanh = sess.run(tf.nn.tanh(x_vals))

# Softsign 激活
print(sess.run(tf.nn.softsign([-1., 0., 1.])))
y_softsign = sess.run(tf.nn.softsign(x_vals))

# Softplus 激活
print(sess.run(tf.nn.softplus([-1., 0., 1.])))
y_softplus = sess.run(tf.nn.softplus(x_vals))

# Exponential linear 激活
print(sess.run(tf.nn.elu([-1., 0., 1.])))

# 绘制不同的激活函数
plt._auto_draw_if_interactive()