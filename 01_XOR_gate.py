import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# xor gate
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [ [0],  [1],  [1],  [0] ]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None,2]) # 2개의 feature
Y = tf.placeholder(tf.float32, [None,1]) # 0과1 하나만 담으므로 1

W = tf.Variable(tf.random_normal([2,1])) # 변수로 선언과 동시에 임의 랜덤값으로 초기값 발생시켜서 저장
b = tf.Variable(tf.random_normal([1]))

# 가설함수
hypothesis = tf.sigmoid(tf.matmul(X,W)+b) # 일차선형함수
# binary
# #은닉층까지는 고려x XOR가 제[대로 분류되는 모델을 ㅏㅊㅈ지 못하는걸 눈으로 확인해보기 위한 실습이다

# 손실함수
loss = -tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
# 텐서플로우의 메소드들 이용 # 선형회귀이므로 MSE

# 경사하강법
train = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) # 계산된 값 >=0.5 --> True 아니면 False(0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})

        if step % 200 == 0:
            print(step, sess.run(loss, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    # Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("H(x) : ", h, ", Correct : ", c, ", Accuracy : ", a)