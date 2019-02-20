# coding:utf-8
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 

seed = 2

def generateds():
	#产生随机数
	rdm = np.random.RandomState(seed)
	X = rdm.randn(300,2)
	#判断条件
	Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
	#遍历数据贴标签
	Y_c = [['red' if y else 'blue'] for y in Y_]
	#整理数据集和标签
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)

	#print(X)
	#print(Y_)
	#print(Y_c)
	plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c))
	plt.show()

	return X, Y_, Y_c