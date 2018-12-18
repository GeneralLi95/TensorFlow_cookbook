#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 07_combining_everything_together.py 
@time: 2018/12/14
本例程中将在鸢尾花数据集上实现一个线性二分类，只区分鸢尾花是山鸢尾或者不是
区分鸢尾花是否为山鸢尾的特征是鸢尾花数据集中的后两个特征,Petal Length and Petal Width
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# 导入鸢尾花数据集
iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target]) # 将数据集中表示类别的整数转成浮点数
iris_2d = np.array([[x[2],x[3]] for x in iris.data])

batch_size = 20

sess = tf.Session()

x1_data  = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1],dtype=tf.float32)

# 模型变量
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# 模型操作
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

# 损失函数 交叉熵
xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = my_output,labels =  y_target)

# 优化函数与变量初始化
my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

# 初始化变量
init = tf.global_variables_initializer()
sess.run(init)

# 开始分类
for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data:rand_x1,x2_data: rand_x2, y_target:rand_y})
    if(i+1)%200 == 0:
        print('Step #' + str(i+1) + 'A = ' + str(sess.run(A)) + ', b = ' + str(sess.run(b)))


# 结果可视化
#
[[slope]] = sess.run(A)
[[intercept]] = sess.run(b)

# 创建分割线
x = np.linspace(0, 3, num=50)
ablineValue = []
for i in x:
    ablineValue.append(slope*i+intercept)

# 在数据散点图上绘制分割线
setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i] == 0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x,non_setosa_y,'ro',label='Non-setosa')
plt.plot(x, ablineValue,'b-')
plt.xlim([0.0,2.7])
plt.ylim([0.0,7.1])
plt.suptitle('Linear Separator For i.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()



