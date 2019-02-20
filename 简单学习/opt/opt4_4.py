#coding:utf-8
#设损失函数 loss=f(w),令W初值为10。反向求最优W
import tensorflow as tf
#定义W初值为10
w = tf.Variable(tf.constant(1, dtype=tf.float32))
#定义损函数loss
loss = tf.square(3*w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
#生成会话，训练40轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(100):	
		sess.run(train_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("After %s steps: w is %f,   loss is %f" % (i, w_val, loss_val))

