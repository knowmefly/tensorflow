# coding:utf-8

# 导入模块
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os
import numpy as np
#import win_unicode_console

#win_unicode_console.enable()

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"


def backward(mnist):
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        forward.IMAGE_SIZE,
        forward.IMAGE_SIZE,
        forward.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])

    # 定义前向传播函数
    y = forward.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    # loss =


    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    # 实例化saver对象
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 初始化所有模型参数
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        # 训练模型
        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
            BATCH_SIZE,
            forward.IMAGE_SIZE,
            forward.IMAGE_SIZE,
            forward.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # sess.run(train_step, feed_dict={x: , y_: })
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
