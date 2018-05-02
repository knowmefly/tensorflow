#coding:utf-8
#设置函数，反向传播求w，即loss最小值
#使用指数衰减学习率，在迭代初期可获得较高下降速度，减小训练轮数
import tensorflow as tf

LEARNING_RATE_BASE = 0.1 #最初学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率 梯度衰减
LEARNING_RATE_STEP = 1 #设置更新轮数 一般：总样本数/BATCH_SIZE

#BATCH_SIZE 计数器，初值为0，设为不训练
global_step = tf.Variable(0, trainable=False)
#定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP, LEARNING_RATE_DECAY, staircase=True)
#定义待优化参数，初值5
w = tf.Variable(tf.constant(5, dtype=tf.float32))
#定义损失函数loss
loss = tf.square(3*w+1)
#定义反向传播方法
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
#生成会话，训练50轮
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(50):
		sess.run(train_step)
		learning_rate_val = sess.run(learning_rate)
		global_step_val = sess.run(global_step)
		w_val = sess.run(w)
		loss_val = sess.run(loss)
		print("Aftre %s steps: global_step is %f, w is %s, learing rate is %f, loss is %f" %(i, global_step_val, w_val, learning_rate_val, loss_val))
