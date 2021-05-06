import numpy as np
import tensorflow as tf


# what are the **kwarg and *arg?
# like def foo(*arg, **kwarg)
# foo(1,2,3,a=1,b=2)
# output is arg = 1,2,3 & kwarg = {'a':1,'b':2}

# over all, one picture generate corresponding one capsule so , the size of one input is batch,H*W*input_N,D,
class Squash(tf.keras.layers.Layer):
    """
        Squash activation function for Efficient capsule neural network

        initial attributes
        -------------------------
        **kwargs
            aims to constructure a dictionary and the purpose of
            in this function
        exp
            is to guarantee denominator is not zero.
        Function
        -------------------------
        call(s)
            use call function to implement the activation method
        get_config()

        compute_output_shape(input_shape)
            keep the s shape not change.
        -------------------------
    """

    def __init__(self, exp=10e-21, **kwargs):
        super(Squash, self).__init__()
        self.exp = exp

    def call(self, s):
        n = tf.norm(s, axis=1, keepdims=True)
        return (1 - 1 / (tf.math.exp(n) + self.exp)) * s / (n + self.exp)

    def get_config(self):
        base_config = super(Squash, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape


class PrimaryCaps(tf.keras.layers.Layer):
    """
        Explanation:
        -------------------------------------
        A depthwise separable convolution with linear activation that preforms just the first step of a depthwise spatial convolution operation.
        Depthwise separable convolution includes the same number of filters as the channel of input. That means one feature map corresponding to one filter which has k*k dimension of weights.
        **** Different from Standard Convolution, k*n output and if a kernel shape of m*m the number of following weights will be increasing less than Standard Convolution because of less feature maps ****

        So assume that we have n channel of output, we gonna have n number of filter and n number of primary capsules.

        Initial Parameters:
        --------------------------------------
        F:
            the number of feature map
        K:
            the size of kernel
        N:
            the number of capsules
            that means the structure of capsules can be changed
        D:
            the dimension of capsules
        s:
          the stride of kernel
    """

    def __init__(self, F, K, N, D, s=1, **kwargs):
        super(PrimaryCaps, self).__init__()
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s

    def build(self, input_shape):
        # for each capsule has FxDx1 for 1 picture
        print('the input shape is', input_shape)
        self.DW_Conv2D = tf.keras.layers.Conv2D(self.F, self.K, self.s, activation='linear', padding='valid',groups=self.F)
        # self.DW_Conv2D = tf.keras.layers.SeparableConv2D(self.F, self.K, self.s, activation='linear', padding='valid',use_bias=False)
        '''self.kernel = self.add_weight(shape=(self.K, self.K, input_shape[-1], None),
                                      initializer='glorot_uniform', name='kernel')
        self.biases = self.add_weight(shape=(None,), initializer='zeros', name='biases')'''
        self.built = True

    def call(self, inputs):
        print("the image shape:", inputs.shape)
        x = self.DW_Conv2D(inputs)
        # x = tf.nn.conv2d(inputs, self.kernel, self.s, 'VALID')
        print(x.shape)
        # this operation is to batch the capsules as NxD, while we have F number of feature maps, in theory, we gonna have F capules and dimension is 1xF
        # now we have NxD that means every D capsules steal into one capsule
        x = tf.keras.layers.Reshape((self.N, self.D))(x)
        x = Squash()(x)

        return x


# begin design self-attention routing part
class FCCaps(tf.keras.layers.Layer):
    """
        Explanation:
        ------------------------------------------
        major steps:
        1. initial input capsules
        2. do the diffraction transformation to lock more feature from the low-level capsules(multiple by weights)
        loop for at least 3 times
        3. do the dynamic routing to update the C(matrix which includes coupling coefficient who contribute more) and classify the high-level capsules
        4. exe sum all capsules multiple by C. why this can be effect, like calculate similarity between two vectors if a feature of capsule can be learned, the value of coupling coefficient will be more larger.
        5. squash activation function
        6. update bij
        7. backward propagation
        :parameter
        -----------------------------------------
            :D
                dimension of primary capsules, every capsules' dimension are the same.
            :N
                number of primary capsules
            :kernel_initializer
                'he_normal' is a normal distribution, sample kernel weight from it.
        optimizer method:
        ------------------------------------------
            :einsuma
                function to do the sigma and easy to formula equation for matrix which has multiple parameters such as [batch_size, channel,height, weight]
    """

    def __init__(self, D, N, kernel_initializer='he_normal', **kwargs):
        super(FCCaps, self).__init__()
        self.D = D
        self.N = N
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]

        # this is hard to understand, you can image it as two sub-MLP transformer
        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D], initializer=self.kernel_initializer,
                                 name='W')
        self.b = self.add_weight(shape=[self.N, input_N, 1], initializer=tf.zeros_initializer(),
                                 name='b')  # bias for high-level capsules
        self.built = True

    def call(self, inputs, training=None):
        # u need be transformed multiple by weights
        # j means input number i means dimension of input(need to discuss)
        u = tf.einsum('...ji,kjiz->...kjz', inputs, self.W)

        c = tf.einsum('...ij,...kj->...i', u, u)[..., None]  # b shape =(None,N,H*W*input_N,1)
        c = c / tf.sqrt(tf.cast(self.D, tf.float32))  # matrix A
        c = tf.nn.softmax(c, axis=1)  # softmax c
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c), axis=-2)  # sum by rows
        v = Squash()(s)
        return v

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D
        }
        base_config = super(FCCaps, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Length(tf.keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), -2) + tf.keras.backend.epsilon())  # sum by columns

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(tf.keras.layers.Layer):
    """

    """

    def call(self, inputs, double_mask=None, **kwargs):

        if type(inputs) is list:
            if double_mask:
                inputs, mask1, mask2 = inputs
                mask1 = tf.transpose(tf.expand_dims(mask1, -1), perm=[0, 2, 1])
                mask2 = tf.transpose(tf.expand_dims(mask2, -1), perm=[0, 2, 1])
            else:
                inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            if double_mask:
                mask1 = tf.keras.backend.one_hot(tf.argsort(x, direction='DESCENDING', axis=-1)[..., 0],
                                                 num_classes=x.get_shape().as_list()[1])
                mask1 = tf.transpose(tf.expand_dims(mask1, -1))

                mask2 = tf.keras.backend.one_hot(tf.argsort(x, direction='DESCENDING', axis=-1)[..., 1],
                                                 num_classes=x.get_shape().as_list()[1])
                mask2 = tf.transpose(tf.expand_dims(mask1, -1))
            else:
                mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if double_mask:
            masked1 = tf.keras.backend.batch_flatten(inputs * mask1)
            masked2 = tf.keras.backend.batch_flatten(inputs * mask2)
            return masked1, masked2
        else:
            masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
