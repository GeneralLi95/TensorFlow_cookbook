#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:yaoli 
@file: 02_layering_nested_operations.py 层嵌套操作
@time: 2018/12/05 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# 新建一个三行五列的数组
my_array = np.array([[1., 3., 5., 7., 9.],
                     [-2., 0., 2., 4., 6.],
                     [-6., -3., 0., 3., 6.]])

x_vals = np.array([my_array, my_array + 1])
# print(x_vals)
x_data = tf.placeholder(tf.float32, shape=(3,5))

# 声明运算中的常量
m1 = tf.constant([[1.], [0.],[-1.], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 声明操作
# 第一层 乘
prod1 = tf.matmul(x_data, m1)
# 第二层 乘
prod2 = tf.matmul(prod1, m2)
# 第三层 加
add1 = tf.add(prod2, a1)

for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data:x_val}))

merged = tf.summary.merge_all(key = 'summaries')

if not os.path.exists('02_logs/'):
    os.makedirs('02_logs/')

my_writer = tf.summary.FileWriter('02_logs/',sess.graph)


