#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 06_batch_stochastic_training.py 批处理和随机训练
@time: 2018/12/11 
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()


# Stochastic Training 随机训练
# 生成数据  np.random.normal(loc=0.0, scale=1.0, sizie=None),均值，标准差，输出的shape
# 我们创建 目标 y size 100的数组，每个值为 10.0

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10.,100)

x_data = tf.placeholder(shape=[1], dtype = tf.float32)
y_target = tf.placeholder(shape=[1], dtype = tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)

# 定义损失函数
loss = tf.square(my_output - y_target)

# 优化方程选择梯度下降法，学习率0.02
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练模型
# 100次迭代，每5次输出一个 A 值和 loss

loss_stochastic = []
for i in range(100):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    sess.run(train_step, feed_dict = {x_data:rand_x, y_target:rand_y})
    if(i+1)%5==0:
        print('Step #' + str(i+1) +' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
        print('Loss = ' + str(temp_loss))
        loss_stochastic.append(temp_loss)

# batch training 批处理
# 重设计算流程  reset the computational graph
ops.reset_default_graph()
sess = tf.Session()

# 声明批大小 Declare batch size
batch_size = 25

# 生成数据 Generate data
x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype = tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype = tf.float32)

# 模型参数与操作
# 我们创建一个参数在图A中 然后我们创建一个模型操作，这个操作仅仅是拿输入数据和 A 相乘

# 创建变量（一个模型参数 A）
A = tf.Variable(tf.random_normal(shape=[1,1]))

# 向图中加入操作
my_output = tf.matmul(x_data, A)

# 损失函数 选择L2损失
loss = tf.reduce_mean(tf.square(my_output - y_target))

# 优化与初始化
init = tf.global_variables_initializer()
sess.run(init)

# 创建优化器，梯度下降法
my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 训练模型
loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data:rand_x,y_target:rand_y})
    if(i+1)%5==0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss, feed_dict={x_data:rand_x, y_target:rand_y})
        print('Loss = ' + str(temp_loss))
        loss_batch.append(temp_loss)

plt.plot(range(0, 100, 5), loss_stochastic, 'b-', label = 'Stochastic Loss')
plt.plot(range(0, 100, 5), loss_batch, 'r--', label = 'Batch Loss, size = 20')
plt.legend(loc='upper right', prop={'size':11})
plt.show()


