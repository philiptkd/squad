import tensorflow as tf

a = tf.constant(1, shape=[3,2,2])
b = tf.constant(1, shape=[1,2,2])
c = tf.matmul(a,b)

print(tf.Session().run(c))
