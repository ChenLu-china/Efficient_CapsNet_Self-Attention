from utils.layer import PrimaryCaps, FCCaps, Length, Mask
import numpy as np
import tensorflow as tf


def efficient_capsnet_graph(input_shape):
    inputs = tf.keras.Input(input_shape)  # size of batch_size, channels(None),36x36x1

    x = tf.keras.layers.Conv2D(32, 5, activation='relu', padding='valid', kernel_initializer='he_normal')(
        inputs)  # (None,32,32,32)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(
        x)  # (None,30,30,64)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        x)  # (None,14, 14, 64)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 3, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        x)  # (None,6, 6, 128)
    x = tf.keras.layers.BatchNormalization()(x)

    x = PrimaryCaps(128, 5, 16, 8, 2)(x)

    digit_caps = FCCaps(10, 16)(x)

    digit_len_caps = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs, outputs=[digit_caps, digit_len_caps], name='Efficient_CapsNet')


def generator_graph(input_shape):
    inputs = tf.keras.Input(10 * 16)

    x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
    x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose):
    """
    :param input_shape:
    :param mode:
    :param verbose:
    :return:
    """
    inputs = tf.keras.Input(input_shape)
    y_true1 = tf.keras.layers.Input(shape=(10,))
    y_true2 = tf.keras.layers.Input(shape=(10,))

    efficient_capsent = efficient_capsnet_graph(input_shape)

    if verbose:
        efficient_capsent.summary()
        print("\n\n")

    digtial_caps, digtial_len_caps = efficient_capsent(inputs)
    masked_by_y1, masked_by_y2 = Mask()([digtial_caps, y_true1, y_true2], double_mask=True)
    masked1, masked2 = Mask()(digtial_caps, double_mask=True)

    generator = generator_graph(input_shape)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train1, x_gen_train2 = generator(masked_by_y1), generator(masked_by_y2)
    x_gen_eval1, x_gen_eval2 = generator(masked1), generator(masked2)

    if mode == 'train':
        return tf.keras.models.Model([inputs, y_true1, y_true2], [digtial_len_caps, x_gen_train1, x_gen_train2],
                                     name='Efficient_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digtial_len_caps, x_gen_eval1, x_gen_eval2],
                                     name='Efficient_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
