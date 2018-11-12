import tensorflow as tf
import numpy as np


emb_size = 4
word_len = 3
qn_len = 2
batch_size = 1
filters = 1
kernel_size = 3
strides = 1


inputs = tf.ones([batch_size, qn_len, word_len, emb_size])
reshaped = tf.reshape(inputs, [batch_size*qn_len, word_len, emb_size])
out = tf.layers.conv1d(reshaped, filters, kernel_size, strides, padding='same', activation=None, use_bias=False, kernel_initializer=tf.ones_initializer)
out = tf.reshape(out, [batch_size, qn_len, word_len, filters])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reshaped, out = sess.run([reshaped, out])
    print(inputs)
    print(reshaped)
    print("kernel_size: ", kernel_size)
    print("strides: ", strides)
    print(out)

"""Use conv1d. Will need to reshape input to have rank 3.
   Need to use stride=embedding size and
   kernel= some multiple of embedding size
   
   [[[[8,8,8,8,8],
      [12,12,12,12,12],
      [8,8,8,8,8]],

     [[14,14,14,14,14],
      [19,19,19,19,19],
      [11,11,11,11,11]]]]

   """
