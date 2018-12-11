#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 05_back_propagation.py 反向传播
@time: 2018/12/06 
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# 一个回归的例子。输入数据是100个随机数，平均值是1.0，标准差是0.1.目标是100个常数10.0
# 我们设置线性回归模型  x_data * A = target_values
# 理论上，我们认为 A 应该等于 10

x_vals = np.random.normal(1,0.1,100)
y_vals = np.repeat(10.,100)
x_data = tf.placeholder(shape = [1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1],dtype=tf.float32)

# 创建模型参数 A
A = tf.Variable(tf.random_normal(shape=[1]))

# 将操作加入到图中
my_output = tf.multiply(x_data, A)

# 将 L2损失函数加入到图中
loss = tf.square(my_output - y_target)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 我们需要创建一个优化操作(optimizing operations),此处我们使用梯度下降优化器(GradientDescentOptimizer),并告诉TensorFlow去最小化
# 损失，这里我们将学习率设置为0.02，学习率决定了学习速度，但是如果学习率太大的话，算法可能不能收敛

my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

# 运行回归流程
for i in range(1400):
    rand_index = np.random.choice(100)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]

    sess.run(train_step, feed_dict={x_data:rand_x, y_target: rand_y})
    if (i+1)%200 == 0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)))
        print('Loss = ' + str())