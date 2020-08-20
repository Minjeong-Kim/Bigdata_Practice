import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# hello
# h:0, e:1, l:2, o:3

# One hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

# RNN input_dim(4) -> output_dim(2), hidden_size:2
hidden_size = 2
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
# print("cell : ", cell.output_size, cell.state_size)

# x_data = np.array([[h]], dtype=np.float32) # x_data에 h라는 데이터 한개만 넣어줌
x_data = np.array([[h,e,l,l,o]], dtype=np.float32) # 배치단위로 5개의 데이터를 넣어줌
# print("x_data:", x_data)

output, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 텐서 활성화

    print("output : ", output.eval())
    # output :  [[[-0.5568914   0.36737362]]] # h=2개라서 output 2개 나옴
    """
    output :  [[[-0.5568914   0.36737362]
                [ 0.70736235  0.02910462]
                [ 0.5458729  -0.84879327]
                [-0.1933814  -0.9233056 ]
                [-0.4347236  -0.19352444]]]
  """