#coding:utf-8

#导入模块
import tensorflow as tf 
import time
from tensorflow.examples.tutorials.mnist import input_data
import forward
import backward
import generateds
import win_unicode_console
win_unicode_console.enable()

TEST_INTERVAL_SECS = 5
REGULARIZER = 0.0001
TEST_NUM = 10000#1 
def test():
	with tf.Graph().as_default() as g:
		#x y_占位
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])

		#前向传播预测结果y
		y = forward.forward(x, REGULARIZER)

		ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		#计算正确率
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

		img_batch, label_batch = generateds.get_tfrecord(TEST_NUM, isTrain=False)#2
		while True:
			with tf.Session() as sess:
				#加载训练好的模型
				ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
				#如果已有ckpt则恢复
				if ckpt and ckpt.model_checkpoint_path:
					#恢复会话
					saver.restore(sess, ckpt.model_checkpoint_path)
					#恢复轮数
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

					coord = tf.train.Coordinator()#3
					threads = tf.train.start_queue_runners(sess=sess, coord=coord)#4
					xs, ys = sess.run([img_batch, label_batch])#5

					#计算准确率
					accuracy_score = sess.run(accuracy, feed_dict={x:xs, y_:ys})
					print("After %s training step(s), test accuracy = %g" %(global_step, accuracy_score))
					
					coord.request_stop()#6
					coord.join(threads)#7
				#如果没有模型
				else:
					print('No checkpoint file found') #模型不存在
					return 
				time.sleep(TEST_INTERVAL_SECS)
def main():
	#加载测试数据集
	# mnist = input_data.read_data_sets("./data/", one_hot=True)
	#调用test（）函数
	test()#8

if __name__=='__main__':
	main()