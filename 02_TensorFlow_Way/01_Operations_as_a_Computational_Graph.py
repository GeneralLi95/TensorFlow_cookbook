#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 01_Operations_as_a_Computational_Graph.py 
@time: 2018/12/05 
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# 创建填入占位符的数据
x_vals = np.array([1., 3., 7., 9.])

# 创建占位符
x_data = tf.placeholder(tf.float32)

# 乘法常量
m = tf.constant(3.)

# 乘法
prod = tf.multiply(x_data, m)
for x_val in x_vals:
    print(sess.run(prod, feed_dict={x_data: x_val}))

# 输出图到tensorboard
merged = tf.summary.merge_all(key='summaries')
if not os.path.exists(('01_logs/')):
    os.makedirs('01_logs/')

my_writer = tf.summary.FileWriter('01_logs', sess.graph)