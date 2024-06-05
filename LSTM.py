# import numpy as np

# class LSTMCell:
#     def __init__(self, input_dim, hidden_dim):
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
        
#         # Xavier initialization for weights
#         self.W_f = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
#         self.b_f = np.zeros((hidden_dim, 1))
        
#         self.W_i = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
#         self.b_i = np.zeros((hidden_dim, 1))
        
#         self.W_c = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
#         self.b_c = np.zeros((hidden_dim, 1))
        
#         self.W_o = np.random.randn(hidden_dim, input_dim + hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
#         self.b_o = np.zeros((hidden_dim, 1))

#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))
    
#     def tanh(self, x):
#         return np.tanh(x)

#     def forward(self, x, h_prev, c_prev):
#         concat = np.concatenate((h_prev, x), axis=0)
        
#         f = self.sigmoid(np.dot(self.W_f, concat) + self.b_f)
#         i = self.sigmoid(np.dot(self.W_i, concat) + self.b_i)
#         c_bar = self.tanh(np.dot(self.W_c, concat) + self.b_c)
#         c = f * c_prev + i * c_bar
#         o = self.sigmoid(np.dot(self.W_o, concat) + self.b_o)
#         h = o * self.tanh(c)
        
#         return h, c

# class BidirectionalLSTM:
#     def __init__(self, input_dim, hidden_dim):
#         self.forward_lstm = LSTMCell(input_dim, hidden_dim)
#         self.backward_lstm = LSTMCell(input_dim, hidden_dim)

#     def forward(self, x):
#         seq_len, input_dim = x.shape
#         h_forward = np.zeros((self.forward_lstm.hidden_dim, seq_len))
#         c_forward = np.zeros((self.forward_lstm.hidden_dim, seq_len))
        
#         h_backward = np.zeros((self.backward_lstm.hidden_dim, seq_len))
#         c_backward = np.zeros((self.backward_lstm.hidden_dim, seq_len))
        
#         h_prev_forward = np.zeros((self.forward_lstm.hidden_dim, 1))
#         c_prev_forward = np.zeros((self.forward_lstm.hidden_dim, 1))
        
#         h_prev_backward = np.zeros((self.backward_lstm.hidden_dim, 1))
#         c_prev_backward = np.zeros((self.backward_lstm.hidden_dim, 1))

#         # Forward LSTM
#         for t in range(seq_len):
#             h_prev_forward, c_prev_forward = self.forward_lstm.forward(x[t].reshape(-1, 1), h_prev_forward, c_prev_forward)
#             h_forward[:, t] = h_prev_forward.ravel()
#             c_forward[:, t] = c_prev_forward.ravel()

#         # Backward LSTM
#         for t in reversed(range(seq_len)):
#             h_prev_backward, c_prev_backward = self.backward_lstm.forward(x[t].reshape(-1, 1), h_prev_backward, c_prev_backward)
#             h_backward[:, t] = h_prev_backward.ravel()
#             c_backward[:, t] = c_prev_backward.ravel()

#         h = np.concatenate((h_forward, h_backward), axis=0)
#         return h

# # Example usage
# input_dim = 10  # Example input dimension
# hidden_dim = 5  # Example hidden dimension
# seq_len = 7     # Example sequence length

# x = np.random.randn(seq_len, input_dim)
# bidirectional_lstm = BidirectionalLSTM(input_dim, hidden_dim)
# output = bidirectional_lstm.forward(x)
# print(output.shape)  # Output shape should be (2*hidden_dim, seq_len)

import tensorflow as tf

class LSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(LSTMCell, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(name='w', shape=(input_shape[-1] + self.units, self.units * 4),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units * 4,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, states):
        h_prev, c_prev = states
        concat = tf.concat([inputs, h_prev], axis=-1)
        z = tf.matmul(concat, self.W) + self.b

        z0, z1, z2, z3 = tf.split(z, num_or_size_splits=4, axis=1)

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        o = tf.sigmoid(z2)
        g = tf.tanh(z3)

        c = f * c_prev + i * g
        h = o * tf.tanh(c)

        return h, [h, c]

class BidirectionalLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BidirectionalLSTM, self).__init__(**kwargs)
        self.units = units
        self.forward_lstm = LSTMCell(units, name="forward_lstm")
        self.backward_lstm = LSTMCell(units, name="backward_lstm")

    def call(self, inputs):
        seq_len = inputs.shape[1]
        batch_size = tf.shape(inputs)[0]
        # batch_size, seq_len, _ = inputs.shape
        h_forward = tf.TensorArray(tf.float32, size=seq_len)
        h_backward = tf.TensorArray(tf.float32, size=seq_len)

        h_prev_forward = tf.zeros((batch_size, self.units))
        c_prev_forward = tf.zeros((batch_size, self.units))

        h_prev_backward = tf.zeros((batch_size, self.units))
        c_prev_backward = tf.zeros((batch_size, self.units))

        # Forward LSTM
        for t in range(seq_len):
            h_prev_forward, [h_prev_forward, c_prev_forward] = self.forward_lstm(inputs[:, t, :], [h_prev_forward, c_prev_forward])
            h_forward = h_forward.write(t, h_prev_forward)

        # Backward LSTM
        for t in reversed(range(seq_len)):
            h_prev_backward, [h_prev_backward, c_prev_backward] = self.backward_lstm(inputs[:, t, :], [h_prev_backward, c_prev_backward])
            h_backward = h_backward.write(t, h_prev_backward)

        h_forward = h_forward.stack()
        h_backward = h_backward.stack()
        h_forward = tf.transpose(h_forward, [1, 0, 2])
        h_backward = tf.transpose(h_backward, [1, 0, 2])

        h_combined = tf.concat([h_forward, h_backward], axis=-1)
        return h_combined
