#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:yaoli 
@file: 04_matrices.py 
@time: 2018/12/03 
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

# 开启图 session
sess = tf.Session()

# 声明矩阵
identity_matrix = tf.diag([1.0, 1.0, 1.0]) # 单位矩阵

# diag 对角矩阵
print(sess.run(identity_matrix))

A = tf.truncated_normal([2, 3])
print(sess.run(A))

B = tf.fill([2, 3], 5.0)
print(sess.run(B))

C = tf.random_uniform([3, 2])
print(sess.run(C))

D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(D))

# 矩阵的操作
print("*" * 20)

# 矩阵加减
print(sess.run(A + B))
print(sess.run(B - B))

# 矩阵乘法
print(sess.run(tf.matmul(B, identity_matrix)))

# transpose 维度交换，如果是二维想到于是转置
print(sess.run(tf.transpose(C)))

# 求矩阵的行列式
print(sess.run(tf.matrix_determinant(D)))

# 求逆矩阵 inverse
print(sess.run(tf.matrix_inverse(D)))

# 求Cholesky分解，指将一个正定的Hermite矩阵分解成一个下三角矩阵与其共轭转置之乘积
print(sess.run(tf.cholesky(identity_matrix)))

# 特征值与特征向量
eigenvalues, eigenvectors = sess.run(tf.self_adjoint_eig(D))
print(eigenvalues)
print(eigenvectors)

