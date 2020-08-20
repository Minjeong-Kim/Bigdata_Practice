import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# Teach hihello : hihell 입력 -> ihello 출력하도록 학습한다.
idx2char = ['h','i','e','l','o']
x_data=[[0, 1, 0, 2, 3, 3]] # 2차원
# one_hot_encoding으로 표현하면 3차원이 될 것이다.
x_one_hot = [[[1,0,0,0,0], # h -> 0
              [0,1,0,0,0], # i -> 1
              [1,0,0,0,0],
              [0,0,1,0,0], # e -> 2
              [0,0,0,1,0], # l -> 3
              [0,0,0,1,0]]]

y_data = [[1, 0, 2, 3, 3, 4]] # ihello
input_dim = 5
hidden_size = 5 # hidden의 사이즈가 5 -> 출력 5
batch_size = 1 # 글자 하나씩 학습시킨다.
sequence_length = 6 # 전체 데이터가 6개니까 time sequence 6개 필요
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim]) # one-hot으로 하면 x_data 3차원 -> [면(총 개수), 행(1개씩 데이터 넣을거), 열]
Y = tf.placeholder(tf.int32, [None, sequence_length]) # 출력은 x 입력 개수만큼 출력되야 된다 (y_data는 2차원)

# RNN model
# cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)

# LSTM model
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
output, _state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# FC layer : 출력의 형태를 펼친다.
X_for_fc = tf.reshape(output, [-1, hidden_size])
"""
fc_w = tf.get_variable("fc_w", [input_dim, 5]) # 5는 임의로 잡아주면 된다.(hideen layer node 수)
fc_b = tf.get_variable("fc_bias", [5])

hypothesis = tf.matmul(X_for_fc, fc_w) + fc_b
"""
hypothesis = tf.contrib.layers.fully_connected(inputs=X_for_fc, num_outputs=hidden_size, activation_fn=None)
    # 이 세팅이 위에 주석처리한것과 같은 동작을 한다.

outputs = tf.reshape(hypothesis, [batch_size, sequence_length, 5])

weights = tf.ones([batch_size, sequence_length]) # 1x6 matrix
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # 만들어진 train으로 텐서를 활성화하면 된다.

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, "loss : ", l, "prediction : ", result, "Y_data : ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
            # squeeze() : 결과값을 하나로 묶어서 하나씩 반환 -> c에 하나씩 담는다.
            # c를 index로 해서 거기에 해당하는 문자(알파벳)를 출력한다.
        print("\n Prediction str:",''.join(result_str))

