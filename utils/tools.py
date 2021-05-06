import tensorflow as tf
import numpy as np


def marginLoss(y_ture, y_pred):
    """
    Explanation:

    :param y_ture:
    :param y_pred:
    :return:
    """
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1

    L = y_ture * tf.square(tf.maximum(0., m_plus - y_pred)) + \
        lbd * (1 - y_ture) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def MultiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred, axis=-1)[:, -2:]
    label_true = tf.argsort(y_true, axis=-1)[:, -2:]

    acc = tf.reduce_sum(tf.cast(label_pred[:, :1] == label_true, tf.int8), axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:, 1:] == label_true, tf.int8), axis=-1)

    return tf.reduce_mean(acc, axis=-1)
