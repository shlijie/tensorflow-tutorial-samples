import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#数据准备
n_observations = 100
xs = np.linspace(-3, 3, n_observations)
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
# xs = np.asarray([1,2,3,4,5,6,7,8,9,10])
# xs = np.asarray([2009,2010,2011,2012,2013,2014,2015,2016,2017,2018])
# ys = np.asarray([0.52,9.36,33.6,191,362,571,912.17,1207,1682,2135])
#plt.scatter(xs, ys)
#plt.show()


#准备好placeholder
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


#初始化参数/权重
W = tf.Variable(tf.random_normal([1]), name='weight')
B = tf.Variable(tf.random_normal([1]), name='bias')
W_2 = tf.Variable(tf.random_normal([1]), name='weight_2')
W_3 = tf.Variable(tf.random_normal([1]), name='weight_3')

#计算预测结果
Y_pred = tf.add(tf.multiply(X, W), B)
Y_pred = tf.add(tf.multiply(tf.pow(X, 2), W_2), Y_pred)
Y_pred = tf.add(tf.multiply(tf.pow(X, 3), W_3), Y_pred)


#计算损失值
sample_num = xs.shape[0]
loss = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / sample_num


#初始化optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


#指定迭代次数，并在session执行graph
n_sample = xs.shape[0]
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('./graphs/polynomial_reg', sess.graph)

    for i in range(600):
        total_loss = 0
        for x, y in zip(xs, ys):
            __, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        if i % 100 == 0:
            print('Epoch{0}:{1}'.format(i, total_loss/n_sample))

    writer.close()

    W, W_2, W_3, B = sess.run([W, W_2, W_3, B])
    # W, W_2, B = sess.run([W, W_2, B])

plt.scatter(xs, ys)
plt.plot(xs, xs**3*W_3 + xs**2*W_2 + xs*W + B, 'r')
# plt.plot(xs, xs**2*W_2 + xs*W + B, 'r')
plt.show()