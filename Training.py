import tensorflow as tf
from Dataset import *

EMOTIONS = ['angry', 'fearful', 'happy', 'sad', 'surprised']

# 定义CNN
def deepnn(x):

  # 向量转为矩阵
  with tf.variable_scope("input"):
    x_image = tf.reshape(x, [-1, 48, 48, 1], name="x_image")

  with tf.variable_scope("Conv1"):
    # conv1
    W_conv1 = weight_variables([5, 5, 1, 64],'Weight_conv1')
    tf.summary.histogram("Conv1" + '/Weight_conv1', W_conv1)

    b_conv1 = bias_variable([64],'Bias_conv1')
    tf.summary.histogram("Conv1" + '/Bias_conv1', b_conv1)

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1,name='h_conv1')

  with tf.variable_scope("pool1"):
    # pool1
    h_pool1 = maxpool(h_conv1)

  with tf.variable_scope("Conv2"):
    # conv2
    W_conv2 = weight_variables([3, 3, 64, 64],'Weight_conv2')
    tf.summary.histogram("Conv2" + '/Weight_conv2', W_conv2)

    b_conv2 = bias_variable([64],'Bias_conv2')
    tf.summary.histogram("Conv2" + '/Bias_conv2', b_conv2)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2,name='h_conv2')

    # 归一化操作
    norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    h_pool2 = maxpool(norm2)

  with tf.variable_scope("Fc1"):
    # Fully connected layer
    W_fc1 = weight_variables([12 * 12 * 64, 384],'Weight_Fc1')
    tf.summary.histogram("Fc1" + '/Weight_Fc1', W_fc1)

    b_fc1 = bias_variable([384],'Bias_Fc1')
    tf.summary.histogram("Fc1" + '/Bias_Fc1', b_fc1)

    h_conv3_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64],name='h_conv3_Fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1,name='h_Fc1')

  with tf.variable_scope("Fc2"):
    # Fully connected layer
    W_fc2 = weight_variables([384, 192],'Weight_Fc2')
    tf.summary.histogram("Fc2" + '/Weight_Fc2', W_fc2)

    b_fc2 = bias_variable([192],'Bias_Fc2')
    tf.summary.histogram("Fc2" + '/Bias_Fc2', b_fc2)

    h_fc2 = (tf.matmul(h_fc1, W_fc2,name='h_Fc2') + b_fc2)

  with tf.variable_scope("Linear"):
    # linear
    W_fc3 = weight_variables([192, 5],'Weight_Fc3')
    tf.summary.histogram("Fc3" + '/Weight_Fc3', W_fc3)

    b_fc3 = bias_variable([5],'Bias_Fc3')
    tf.summary.histogram("Fc3" + '/Bias_Fc3', b_fc3)

  with tf.variable_scope("result"):
    y_conv = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3,name='output_y')
    tf.summary.histogram("Result" + '/output_y', y_conv)

  return y_conv

# 卷积操作
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 最大下采样操作(即池化操作)
def maxpool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')
  #返回池化区域最大值

#权重定义函数
def weight_variables(shape,name):
  initial = tf.truncated_normal(shape, stddev=0.1) ##生成正态分布的变量，且这个函数产生的随机数与均值的差距不会超过两倍的标准差
  return tf.Variable(initial,name=name)

#偏置定义函数
def bias_variable(shape,name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial,name=name)

# 图像到张量的转换函数
def image_to_tensor(image):
  tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
  return tensor

def train():
  fer2013 = input_data('./data/fer2013/fer2013.csv')
  max_train_steps = 30001
  with tf.variable_scope("input_x"):
    x = tf.placeholder(tf.float32, [None, 2304],name="input_x")
  with tf.variable_scope("Default_y"):
    y_ = tf.placeholder(tf.float32, [None, 5], name="Default_y")

  y_conv = deepnn(x)

  #定义交叉熵
  with tf.variable_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #学习率=1e-4

  with tf.variable_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) #预测正确的标签
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

  with tf.Session() as sess:
    #创建模型保存器
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("./logdir", sess.graph)

    for step in range(max_train_steps):
      batch = fer2013.train.next_batch(50)
      if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        print('step %d: training accuracy %g' % (step, train_accuracy))

      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

      if step + 1 == max_train_steps:
        saver.save(sess, './models/emotion_model', global_step=step + 1)
        print('step finish!')

      if step % 1000 == 0:
        testacc=accuracy.eval(feed_dict={x: fer2013.validation.images, y_: fer2013.validation.labels})
        tf.summary.scalar('testacc', testacc)
        print('Test accuracy is %g' % testacc)

      if step % 50 == 0:
        result = sess.run(merged, feed_dict={x: batch[0], y_: batch[1]})
        # result2=sess.run(testacc, feed_dict={x: fer2013.validation.images, y_: fer2013.validation.labels})

      train_writer.add_summary(result, step)
      # train_writer.add_summary(result2, step)
    train_writer.close()

if __name__ == '__main__':
  tf.app.run(main=train)
