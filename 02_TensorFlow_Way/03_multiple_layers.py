#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:yaoli 
@file: 03_multiple_layers.py 多层网络
@time: 2018/12/05 
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# 创建一个小图像，像素尺寸为 4 * 4 , 然后通过多层网络传播

x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

x_data = tf.placeholder(tf.float32, shape=x_shape)

# 第一层，一个空间移动窗口，大小是 2 * 2

my_filter = tf.constant(0.25, shape = [2,2,1,1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_Avg_Window')

# 第二层，自定义层
# squeeze 删除所有大小是1的维度
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.],[-1.,3.]])
    b = tf.constant(1., shape=[2,2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b) # Ax + b
    return (tf.sigmoid(temp))

# 把自定义层加到图中
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

print(sess.run(mov_avg_layer, feed_dict={x_data:x_val}))

print(sess.run(custom_layer1,feed_dict={x_data:x_val}))
# 保存日志以便在Tensorboard里查看
merged = tf.summary.merge_all(key='summaries')

if not os.path.exists('03_logs/'):
    os.makedirs('03_logs/')

my_writer = tf.summary.FileWriter('03_logs/', sess.graph)

