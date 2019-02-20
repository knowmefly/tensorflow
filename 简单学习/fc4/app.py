#coding:utf-8
import tensorflow as tf 
import numpy as np
from PIL import Image 
import forward
import backward
import win_unicode_console
win_unicode_console.enable()
REGULARIZER = 0.0001

#图片预处理
def pre_pic(picName):
	img = Image.open(picName)
	reIm = img.resize((28,28), Image.ANTIALIAS)
	im_arr = np.array(reIm.convert('L'))
	threshold = 50
	for i in range(28):
		for j in range(28):
			im_arr[i][j] = 255 - im_arr[i][j]
			if (im_arr[i][j] < threshold):
				im_arr[i][j] = 0
			else: im_arr[i][j] =255

	nm_arr = im_arr.reshape([1,784])
	nm_arr = nm_arr.astype(np.float32)
	img_ready = np.multiply(nm_arr, 1.0/255.0)

	return img_ready

def restore_model(testPicArr):
	#创建默认图
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
		y = forward.forward(x, REGULARIZER)
		#概率最大预测值
		preValue = tf.argmax(y, 1) 

		#实现滑动平均，控制更新进度
		variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
		variable_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variable_to_restore)

		with tf.Session() as sess:
			#通过checkpoint定位最新保存模型
			ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1

def application():
	testNum = int(input("input the number of test pictures:"))
	for i in range(testNum):
		testPic = input("the path of test picture:")
		#对图片预处理
		testPicArr = pre_pic(testPic)
		#将符合要求的数据喂入模型
		preValue = restore_model(testPicArr)
		print ("The prediction_model is :", preValue)

def main():
	application()

if __name__=='__main__':
	main()
