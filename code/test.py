import tensorflow as tf

a = tf.get_variable("a", [2,3], tf.float32)
b = a*3
c = tf.tile(b, [2, 3])

collection = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
for item in collection:
    print(item)
