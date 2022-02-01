import tensorflow as tf


@tf.function
def custom_acc(y_true, y_pred):
    if len(tf.shape(y_pred)) > 2:
        y_pred  = tf.nn.softmax(y_pred)[:,-1,:]
    else:
        y_pred  = tf.nn.softmax(y_pred)

    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
    y_pred = tf.expand_dims(y_pred, 1)

    correct = tf.math.equal(y_true, y_pred)
    correct = tf.cast(correct, tf.float32)

    return tf.reduce_mean(correct)

def custom_r2(y_true, y_pred):
    true = tf.slice(y_true, [0,0,0], [-1,-1,1])
    true = tf.squeeze(true)
    mask = tf.slice(y_true, [0,0,1], [-1,-1,1])
    mask = tf.squeeze(mask)
    y_pred = tf.squeeze(y_pred)

    unexplained_error = tf.square(true - y_pred)
    unexplained_error = tf.reduce_sum(unexplained_error*mask, 1)

    true_mean = tf.expand_dims(tf.reduce_mean(true, 1), 1)
    total_error = tf.square(true-true_mean)
    total_error = tf.reduce_sum(total_error*mask, 1)

    r_squared = 1-tf.math.divide_no_nan(unexplained_error, total_error)

    return tf.reduce_mean(r_squared)
