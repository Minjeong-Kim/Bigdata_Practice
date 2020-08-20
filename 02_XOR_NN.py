import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# xor gate
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [ [0],  [1],  [1],  [0] ]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None,2])
Y = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.random_normal([2]))
layer1 = tf.sigmoid(tf.matmul(X,W1)+b1)

W2 = tf.Variable(tf.random_normal([2,1]))
b2 = tf.Variable(tf.random_normal([1]))

# 가설함수 : 최종 출력단 함수
hypothesis = tf.sigmoid(tf.matmul(layer1,W2)+b2) 

# 손실함수
loss = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
# 텐서플로우의 메소드들 이용 # 선형회귀이므로 MSE

# 경사하강법
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}), sess.run([W1, W2])) # 은닉층이 있으니까 값 2개

    # Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("H(x) : ", h, ", Correct : ", c, ", Accuracy : ", a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}), sess.run([W1, W2])) # 은닉층이 있으니까 값 2개

    # Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("H(x) : ", h, ", Correct : ", c, ", Accuracy : ", a)