#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 01_lin_reg_inverse.py 
@time: 2018/12/18
线性回归 逆矩阵法
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

# 产生数据
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)
# print("x_vals",'\n',x_vals)
# print('*'*80)
# print(y_vals)

# 设计矩阵 A
x_vals_column = np.transpose((np.matrix(x_vals)))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# print('x_vals_column\n',x_vals_column)
# print(A)

# y 矩阵
y = np.transpose(np.matrix(x_vals))
A_tensor = tf.constant(A)
y_tensor = tf.constant(y)

# 开始计算TensorFlow操作的参数
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, y_tensor)

# 计算结果，提取系数
solution_eval = sess.run(solution)
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

# 打印结果，绘制拟合曲线
print('slope: ' + str(slope))
print('y_intercept: ' + str(y_intercept))

best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)

# 使用 matplotlib 绘图
plt.plot(x_vals, y_vals, 'o', label = 'Data')
plt.plot(x_vals, best_fit, 'r-' ,label = 'Best fit line', linewidth = 3)
plt.legend(loc='upper left')
plt.show()

