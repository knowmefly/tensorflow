#!/usr/bin/python
# coding:utf-8

import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68]  #样本RGB的平均值


class Vgg16():
	def __init__(self, vgg16_path=None):
		if vgg16_path is None:
			vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")  #os.getcwd() 返回当前目录
			self.data_dict = np.load(vgg16_path, encoding='latin1').item()  #遍历键值对，导入模型参数

		for x in self.data_dict:
			print(x)

	def forward(self, images):

		print("build model started")
		start_time = time.time()  #获取前向传播的开始时间
		rgb_scaled = images * 255.0  #初始化操作
		#从RGB转换色彩通道到BGR
		red, green, blue = tf.split(rgb_scaled, 3, 3)
		bgr = tf.concat([
			blue - VGG_MEAN[0],
			green - VGG_MEAN[1],
			red - VGG_MEAN[2]], 3)

		#5段卷积、池化 3层全连接
		self.conv1_1 = self.conv_layer(bgr, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

		self.fc6 = self.fc_layer(self.pool5, "fc6")  #根据命名空间做加权求和运算
		self.relu6 = tf.nn.relu(self.fc6)  #经过relu激活函数

		self.fc7 = self.fc_layer(self.relu6, "fc7")
		self.relu7 = tf.nn.relu(self.fc7)

		self.fc8 = self.fc_layer(self.relu7, "fc8")  #做softmax分类，得到各类别的概率
		self.prob = tf.nn.softmax(self.fc8, name="prob")

		end_time = time.time()  #前向传播结束时间
		print(("time consuming: %f" % (end_time - start_time)))

		self.data_dict = None  #清空本次词典

	#定义卷积运算
	def conv_layer(self, x, name):
		with tf.variable_scope(name):  #根据命名空间找到对应卷基层网络参数
			w = self.get_conv_filter(name)  #读到该层的卷积核
			conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')  #卷积计算
			conv_biases = self.get_bias(name)  #读到偏置项
			result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))  #加上偏置，做激活运算
			return result
	#定义获取卷积核的函数
	def get_conv_filter(self, name):
		return tf.constant(self.data_dict[name][0], name="filter")
	#定义获取偏执项的函数
	def get_bias(self, name):
		return tf.constant(self.data_dict[name][1], name="biases")
	#定义最大池化操作
	def max_pool_2x2(self, x, name):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
	#定义全连接层的前向传播计算
	def fc_layer(self, x, name):
		with tf.variable_scope(name):  #根据命名空间name做全连接层的计算
			shape = x.get_shape().as_list()  #获取该层的维度信息列表
			dim = 1
			for i in shape[1:]:
				dim *= i  #将每层的维度相乘
			# 改变特征图的形状,也就是将得到的多维特征做拉伸操作,只在进入第六层全连接层做该操作
			x = tf.reshape(x, [-1, dim])  # 读到权重值
			w = self.get_fc_weight(name)  # 读到偏置项值
			b = self.get_bias(name)

			result = tf.nn.bias_add(tf.matmul(x, w), b)  # 对该层输入做加权求和,再加上偏置
			return result

	# 定义获取权重的函数
	def get_fc_weight(self, name):  # 定义获取权重的函数
		return tf.constant(self.data_dict[name][0], name="weights")

