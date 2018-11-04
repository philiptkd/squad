import tensorflow as tf

a = tf.constant(1, shape=[3,4,3])
b = a[:,:-1,:]
print(tf.Session().run(b))
