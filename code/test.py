import tensorflow as tf

# N = 3, M = 2, h = 1
# S_ij = w.T[c_i; q_j; c_i*q_j]
c = tf.constant([[[1,2,3,4],[2,3,4,5],[3,4,5,6]]])   # (batch_size=1, N, 2h)
q = tf.constant([[[5,6,7,8],[7,8,9,10]]]) # (batch_size=1, M, 2h)

c = tf.tile(tf.expand_dims(c, 2), [1, 1, 2, 1])
q = tf.tile(tf.expand_dims(q, 1), [1, 3, 1, 1])
S = tf.concat([c, q, c*q], axis=3)


w = tf.ones(12,dtype=tf.int32) # shape (4*h,)
# S*w: (batch_size, N, M, 4*h)x(batch_size, N, 4*h, 1)
w = tf.reshape(w, [1,1,12,1])
w = tf.tile(w, [1,3,1,1])

T = tf.squeeze(tf.matmul(S,w), -1)
shape = tf.shape(T)

c,q,S,T,shape = tf.Session().run([c,q,S,T,shape])

print(c,q,S)

#a = tf.constant([[[1,1],[20,20]], [[3,3],[4,4]]], dtype=tf.float32)
#m = tf.reduce_max(a, -1)
#print(tf.Session().run(m))
