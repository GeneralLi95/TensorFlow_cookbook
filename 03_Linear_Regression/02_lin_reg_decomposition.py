#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:yaoli 
@file: 02_lin_reg_decomposition.py 
@time: 2018/12/27
使用 Cholesky 分解求线性回归
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

sess = tf.Session()

x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)


# 建立设计矩阵 Create design matrix
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# 建立y矩阵 Create y matrix
y = np.transpose(np.matrix(y_vals))

# 建立张量 Create tensors
A_tensor = tf.constant(A)
y_tensor = tf.constant(y)

# 开始进行 Cholesky 分解，Cholesky decomposition
# Find Cholesky Decomposition
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)

# Solve L * y = t(A) * b
tA_y = tf.matmul(tf.transpose(A_tensor), y)
sol1 = tf.matrix_solve(L, tA_y)

# Solve L' * y = sol1
sol2 = tf.matrix_solve(tf.transpose(L), sol1)

solution_eval = sess.run(sol2)

# 提取系数 Extract coefficients
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]

print('Slope:' + str(slope))
print('y_intercept:' + str(y_intercept))

# 最佳拟合曲线 Get best fit line
best_fit = []
for i in x_vals:
    best_fit.append(slope*i + y_intercept)

# 绘制结果 Plot the results
plt.plot(x_vals, y_vals, 'o', label = 'Data')
plt.plot(x_vals, best_fit, 'r-', label = 'Best fit line', linewidth = 3)
plt.legend(loc='upper left')
plt.show()

