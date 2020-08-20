"""
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # 4차원 [면, 행, 열, 채널]
    # mnist의 원래 사이즈인 28x28로 복원 시키면서 채널은 흑백이니까 1로 지정
    # 면은 -1로 지정하면 알아서 남은 수만큼 지정(-1에는 배치사이즈가 100이니까 100이 들어갈 것이다.)
Y = tf.placeholder(tf.float32, [None, 10])

#### Layer 1
# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#   conv  -> (?, 28, 28, 32)
# pooling -> (?, 14, 14, 32) ; 2x2로 사이즈 지정

L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    #   conv  -> (?, 28, 28, 32)
    # conv2d(이미지, 필터, stride모양, padding크기)
    # stride는 얼마나 이동하면서 convolution을 구할것인가.
    # padding은 주변에 얼마나 이미지 주변에 패딩 크기를 추가할 것인가.
    # stride와 padding을 통해 우리가 원하는 크기로 이미지를 축소할 수 있다.
    # padding='SAME'으로 지정하면 이미지가 축소되지 않고 원래 크기로 출력될 수 있도록 자동으로 패딩을 추가하라는 뜻
    # stride=[1,1,1,1] 이미지가 4차원이니까 한칸씩 이동한다.
L1 = tf.nn.relu(L1) # activation func : relu
tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # pooling -> (?, 14, 14, 32)
    # Pooling을 통해 또 출력 이미지 사이즈를 조절할 수 있다.

#### Layer 2
# L2 ImgIm shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01)) # 최종 출력 크기를 64로 만들어준다.
#   conv  -> (?, 14, 14, 64)
# pooling -> (?, 7, 7, 64) ; 2x2로 사이즈 지어

L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #   conv  -> (?, 14, 14, 64)
    # 출력결과를 64 크기로 출력
L2 = tf.nn.relu(L2) # activation func : relu

L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    # pooling -> (?, 7, 7, 64) ; 2x2로 사이즈

L2_flat = tf.reshape(L2, [-1, 7*7*64]) # 출력 사이즈를 고정시켜서 반환한다. ( 7*7*64 임의로 지정한것)

# Final FC 7*7*64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape = [7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))

# 가설함수
Hx = tf.matmul(L2_flat, W3) + b

# 손실함수
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Hx, labels=Y))

# 학습 알고리즘 -- Adam optimizer 이용
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

#### 텐서 활성화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs): # 15번 반복
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size) # 550번 (batch_size=100)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([loss, train], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch+1), 'cost=','{:.9f}'.format(avg_cost))

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(Hx ,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
feed = {X:mnist.test.images, Y:mnist.test.labels}
print('Accuracy:',sess.run(accuracy,feed_dict=feed))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction:", sess.run(tf.argmax(Hx, 1),
                              feed_dict={X:mnist.test.images[r:r+1]}))
plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys')
plt.show()

sess.close()

"""
import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
#       conv        ->  (?, 28, 28, 32)
#       pooling     ->  (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# L2 ImaIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
#       conv        ->  (?, 14, 14, 64)
#       pooling     ->  (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1,W2,strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# final FC (fully-connected) 7*7*64 input -> 10 outputs
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))



# 가설함수
hypothesis = tf.matmul(L2_flat, W3) + b

# 손실함수
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

# 알고리즘
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([loss, train], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch+1), 'cost=','{:.9f}'.format(avg_cost))

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
feed = {X:mnist.test.images, Y:mnist.test.labels}
print('Accuracy:',sess.run(accuracy,feed_dict=feed))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction:", sess.run(tf.argmax(hypothesis,1),
                              feed_dict={X:mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys')
plt.show()

sess.close()