import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, shape=[None,784])
Y = tf.placeholder(tf.float32, shape=[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))

train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost) # Adam optimizer 이용

sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epochs = 15
batch_size = 100 # 1번에 처리할 데이터 크기

for epoch in range(training_epochs): # 15번 반복
    avg_cost = 0 # 손실 평균값을 avg_cost 변수에 누적
    total_batch = int(mnist.train.num_examples / batch_size)
        # mnist.train.num_examples : 55000개 훈련 데이터셋에 대한 정보를 가지고 잇다.
        # total_batch = 55000/100 = 550번 = 1 epoch
        # 총 학습 수는 15번
    for i in range(total_batch): # 550번 반복
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost, train], feed_dict=feed_dict) # 학습이 배치사이즈 100단위로 처리된다. (550번 반복)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch+1), 'cost=','{:.9f}'.format(avg_cost)) # 1 epoch가 끝날때마다 총 cost를 출력하게 한다 -> 총 15번 출력된다.

print('Learning Finished!')

correct_prediction = tf.equal(tf.argmax(hypothesis,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

feed = {X:mnist.test.images, Y:mnist.test.labels}
print('Accuracy:',sess.run(accuracy,feed_dict=feed))

r = random.randint(0, mnist.test.num_examples-1)
print("Label:",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction:", sess.run(tf.argmax(hypothesis,1),
                              feed_dict={X:mnist.test.images[r:r+1]}))

plt.imshow(mnist.test.images[r:r+1].reshape(28,28),
           cmap='Greys', interpolation='nearest')
plt.show()





