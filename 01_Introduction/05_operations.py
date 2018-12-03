#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author:yaoli 
@file: 05_operations.py 
@time: 2018/12/03 
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Open graph session
sess = tf.Session()

# 代数操作
print(sess.run(tf.div(3, 4)))
print(sess.run(tf.truediv(3, 4)))
print(sess.run(tf.floordiv(3.0, 4.0)))

# 取模
print(sess.run(tf.mod(22.0, 5.0)))
# 叉乘
print(sess.run(tf.cross([1., 0., 0.], [0., 1., 0.])))

# 三角函数
print(sess.run(tf.sin(3.1416)))
print(sess.run(tf.cos(3.1416)))
print(sess.run(tf.div(tf.sin(3.1416 / 4.), tf.cos(3.1416 / 4.))))

# 自定义运算 f(x) = 3 * x ^2 - x + 10

test_nums = range(15)


def custom_polynomial(x_val):
    return (tf.subtract(3 * tf.square(x_val), x_val) + 10)


print(sess.run(custom_polynomial(11)))
expected_output = [3 * x * x - x + 10 for x in test_nums]
print(expected_output)

for num in test_nums:
    print(sess.run(custom_polynomial(num)))

