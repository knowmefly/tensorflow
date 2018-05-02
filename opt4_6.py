#coding:utf-8

import tensorflow as tf

#  定义变量及滑动平均类
#定义变量w1
w1 = tf.Variable(0, dtype=tf.float32)
#定义轮数
global_step = tf.Variable(0, trainable=False)
#实例化滑动平均数
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

#汇总列表
ema_op = ema.apply(tf.trainable_variables())

#  查看迭代变化值
with tf.Session() as sess:
	#初始化
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	#获取多个w1 和 w1滑动平均值
	print(sess.run([w1, ema.average(w1)]))
	
	#w1赋值为1 
	sess.run(tf.assign(w1, 1))
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))
	
	#更新step和w1的值，模拟一百轮
	sess.run(tf.assign(global_step, 100))
	sess.run(tf.assign(w1, 10))
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))

	#每次sess.run会更新一次w1
	sess.run(ema_op)
	print(sess.run([w1, ema.average(w1)]))
	
