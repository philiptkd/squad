# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob, cell):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        if cell == "GRU":
            self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
            self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        elif cell == "LSTM":
            self.rnn_cell_fw = rnn_cell.BasicLSTMCell(self.hidden_size)
            self.rnn_cell_bw = rnn_cell.BasicLSTMCell(self.hidden_size)

        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist

class BiDAF(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size, num_values):
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
	self.num_values = num_values

        """Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)"""
    def build_graph(self, values, values_mask, keys):
        with vs.variable_scope("BiDAF"):
            batch_size = tf.shape(values)[0]
            num_values = tf.shape(values)[1]
            num_keys = tf.shape(keys)[1]
            
            # give both shape (batch_size, num_values, num_keys, value_vec_size)
            repeated_values = tf.tile(tf.expand_dims(values, 2), [1, 1, num_keys, 1]) 
            repeated_keys = tf.tile(tf.expand_dims(keys, 1), [1, num_values, 1, 1])

            # unweighted similarity matrix. shape (batch_size, num_values, num_keys, 3*value_vec_size)
            unweighted = tf.concat([repeated_values, repeated_keys, repeated_values*repeated_keys], axis=3)
           
            # shape (batch_size, num_values, 3*value_vec_size, 1)
            weights = tf.get_variable("weights", [1, self.num_values, 3*self.value_vec_size, 1], tf.float32)
            weights = tf.tile(weights, [batch_size, 1, 1, 1])

            # multiply to get similarity matrix logits. shape (batch_size, num_values, num_keys)
            attn_logits = tf.squeeze(tf.matmul(unweighted, weights), -1)

            # get values mask for logits
            attn_logits_mask = tf.reshape(values_mask, [batch_size, num_values, 1])

            # apply mask and softmax. shape (batch_size, num_values, num_keys)
            _, attn_dist_c2q = masked_softmax(attn_logits, attn_logits_mask, 2)

            # Use attention distribution to take weighted sum of values
            a_c2q = tf.matmul(tf.transpose(attn_dist_c2q, perm=[0,2,1]), values) # shape (batch_size, num_keys, value_vec_size)

            # mask before getting max so padding doesn't affect reduction. shape (batch_size, num_keys)
            max_per_row = tf.reduce_max(attn_logits*tf.cast(attn_logits_mask, 'float'), 1)
            
            # since padding was already masked, none will be left after reduce_max. can do regular softmax. shape (batch_size, num_keys)
            beta = tf.nn.softmax(max_per_row, dim=-1)
            a_q2c = tf.matmul(tf.expand_dims(beta, 1), keys)    # shape (batch_size, 1, value_vec_size)
            a_q2c = tf.tile(a_q2c, [1, num_keys, 1])    # to make it the same shape as a_c2q: (batch_size, num_keys, value_vec_size)
            

            # Apply dropout
            a_c2q = tf.nn.dropout(a_c2q, self.keep_prob) # both shape (batch_size, num_keys, value_vec_size)
            a_q2c = tf.nn.dropout(a_q2c, self.keep_prob)

            return a_c2q, a_q2c

class CoAttn(object):
    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        with vs.variable_scope("CoAttn"):
            batch_size = tf.shape(values)[0]
            num_values = tf.shape(values)[1]
            num_keys = tf.shape(keys)[1]

            # same shape as q/values: (batch_size, num_values, value_vec_size)
            q_prime = tf.contrib.layers.fully_connected(values, num_outputs=self.value_vec_size, activation_fn=tf.tanh)
        
            # create sentinels c0 and q0
            c0 = tf.get_variable("c0_sentinel", [1, 1, self.value_vec_size], dtype=tf.float32)
            q0 = tf.get_variable("q0_sentinel", [1, 1, self.value_vec_size], dtype=tf.float32)
            c0 = tf.tile(c0, [batch_size, 1, 1]) # to share weights between examples in the batch
            q0 = tf.tile(q0, [batch_size, 1, 1])

            c = tf.concat([keys, c0], axis=1) # shape (batch_size, num_keys + 1, value_vec_size)
            q_prime = tf.concat([q_prime, q0], axis=1) # shape (batch_size, num_values + 1, value_vec_size)

            # create affinity matrix
            L = tf.matmul(c, tf.transpose(q_prime, perm=[0,2,1])) # shape (batch_size, num_keys + 1, num_values + 1)
            
            # create mask for affinity matrix
            L_mask = tf.fill([batch_size, 1], 1) # creates tensor of 1s with shape (batch_size, 1)
            L_mask = tf.concat([values_mask, L_mask], 1) # shape (batch_size, num_values + 1)
            L_mask = tf.expand_dims(L_mask, axis=1) # shape (batch_size, 1, num_values + 1)


            # get c2q attention (for a given context word, how similar is each question word)
            _, alpha = masked_softmax(L, L_mask, -1) # shape (batch_size, num_keys+1, num_values+1)
            a = tf.matmul(alpha, q_prime) # shape (batch_size, num_keys+1, value_vec_size)

            # get q2c attention (for a given question word, how similar is each context word)
            _, beta = masked_softmax(L, L_mask, 1) # shape(batch_size, num_keys+1, num_values+1)
            b =  tf.matmul(tf.transpose(beta, perm=[0,2,1]), c) # shape (batch_size, num_values+1, value_vec_size)

            # second-level attention
            s = tf.matmul(alpha, b) # shape (batch_size, num_keys+1, value_vec_size)

            # biLSTM
            LSTM_encoder = RNNEncoder(2*self.value_vec_size, self.keep_prob, "LSTM")
            LSTM_input = tf.concat([s,a], axis=-1) # shape (batch_size, num_keys+1, 2*value_vec_size)
            LSTM_input = LSTM_input[:,:-1,:] # remove sentinels. shape (batch_size, num_keys, 2*value_vec_size)
            LSTM_mask = tf.fill([batch_size, num_keys], 1)
            u = LSTM_encoder.build_graph(LSTM_input, LSTM_mask) # shape (batch_size, num_keys, 4*value_vec_size)
            
            # dropout is applied within the LSTM

            return u

class SelfAttn(object):
    def __init__(self, keep_prob, num_keys, encoding_size):
        self.keep_prob = keep_prob
        self.enc_size = encoding_size
        self.N = num_keys

    def build_graph(self, in_encodings):    # in_encodings shape is (batch_size, N, enc_size)
        with vs.variable_scope("SelfAttn"):
	    self.B = tf.shape(in_encodings)[0]            

	    # tile along different axes and then concatenate to get every possible combination of elements of original tensor
            # has the unfortunate downside of requiring a lot of memory
            u1 = tf.tile(tf.expand_dims(in_encodings, 1), [1, self.N, 1, 1]) # shape (batch_size, N, N, enc_size)
            u2 = tf.tile(tf.expand_dims(in_encodings, 2), [1, 1, self.N, 1]) # shape (batch_size, N, N, enc_size)
            u = tf.concat([u1, u2], -1) # shape (batch_size, N, N, 2*enc_size)

            # shape (batch_size, N, 2*enc_size, 2*enc_size)
            W = tf.tile(tf.get_variable("W", [1, self.N, 2*self.enc_size, 2*self.enc_size]), [self.B, 1, 1, 1])
            
            # shape (batch_size, N, N, 2*enc_size)
            z = tf.tanh(tf.matmul(u, W))
            z_drop = tf.nn.dropout(z, keep_prob=self.keep_prob)

            # shape (batch_size, N, 2*enc_size, 1)
            V = tf.tile(tf.get_variable("V", [1, self.N, 2*self.enc_size, 1], dtype=tf.float32), [self.B, 1, 1, 1])

            S = tf.squeeze(tf.matmul(z_drop, V), -1) # shape (batch_size, N, N)
            a = tf.nn.softmax(S,2) # shape (batch_size, N, N)
            c = tf.matmul(tf.transpose(a, perm=[0,2,1]), in_encodings) # shape (batch_size, N, enc_size)
            
            attn_enc = tf.concat([in_encodings, c], -1) # shape (batch_size, N, 2*enc_size)

            # shape (batch_size, 2*enc_size, 2*enc_size)
            Wg = tf.tile(tf.get_variable("Wg", [1, 2*self.enc_size, 2*self.enc_size], tf.float32), [self.B, 1, 1])

            gate = tf.sigmoid(tf.matmul(attn_enc, Wg)) # shape (batch_size, N, 2*enc_size)
            GRU_input = gate*attn_enc # shape (batch_size, N, 2*enc_size)
            GRU_encoder = RNNEncoder(self.enc_size, self.keep_prob, "GRU")
            GRU_mask = tf.fill([self.B, self.N], 1) # shape (batch_size, N)
            h = GRU_encoder.build_graph(GRU_input, GRU_mask) # shape (batch_size, N, enc_size*2)

            # dropout is applied within the GRU
            
            return h


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
