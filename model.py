import tensorflow as tf
from ops import *
from utils import *


d_bn2 = batch_norm(name='d_bn2')
d_bn3 = batch_norm(name='d_bn3')
d_bn4 = batch_norm(name='d_bn4')

g_bn0 = batch_norm(name='g_bn0')
g_bn1 = batch_norm(name='g_bn1')
g_bn2 = batch_norm(name='g_bn2')
g_bn3 = batch_norm(name='g_bn3')
g_bn4 = batch_norm(name='g_bn4')
g_bn5 = batch_norm(name='g_bn5')
g_bn6 = batch_norm(name='g_bn6')
g_bn7 = batch_norm(name='g_bn7')


def generator(z, label=None, batch_size=64):
    with tf.variable_scope("generator") as scope:

        label_conv = tf.reshape(label, [batch_size, 1, 1, 1])
        z = concat([z, label], 1)

        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        h1 = conv_cond_concat(h1, label_conv)

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      5, 5, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))
        h2 = conv_cond_concat(h2, label_conv)

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      5, 5, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))
        h3 = conv_cond_concat(h3, label_conv)

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      5, 5, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))
        h4 = conv_cond_concat(h4, label_conv)

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      5, 5, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        h5 = conv_cond_concat(h5, label_conv)

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      5, 5, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))
        h6 = conv_cond_concat(h6, label_conv)

        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        h7 = conv_cond_concat(h7, label_conv)

        h8 = deconv2d(h7, [batch_size, 128, 128, 3],
                      5, 5, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8


def discriminator(image, label=None, batch_size=64, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()

        label_conv = tf.reshape(label, [batch_size, 1, 1, 1])
        x = conv_cond_concat(image, label_conv)

        h1 = conv2d(x, 64, 5, 5, 2, 2, name='d_h1_conv')
        h1 = lrelu(h1)
        h1 = conv_cond_concat(h1, label_conv)

        h2 = conv2d(h1, 128, 5, 5, 2, 2, name='d_h2_conv')
        h2 = lrelu(d_bn2(h2))
        h2 = conv_cond_concat(h2, label_conv)

        h3 = conv2d(h2, 256, 5, 5, 2, 2, name='d_h3_conv')
        h3 = lrelu(d_bn3(h3))
        h3 = conv_cond_concat(h3, label_conv)

        h4 = conv2d(h3, 512, 5, 5, 2, 2, name='d_h4_conv')
        h4 = lrelu(d_bn4(h4))
        h4 = conv_cond_concat(h4, label_conv)
        h4 = tf.reshape(h4, [batch_size, -1])

        h5 = linear(h4, 1024, 'd_h5_lin')
        h5 = lrelu(h5)
        h5 = concat([h5, label], 1)

        h6 = linear(h5, 1, 'd_h6_lin')

        return tf.nn.sigmoid(h6), h6


def sampler(z, label=None, batch_size=64):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        label_conv = tf.reshape(label, [batch_size, 1, 1, 1])
        z = concat([z, label], 1)

        # fully-connected layers
        h1 = linear(z, 256 * 8 * 8, 'g_h1_lin')
        h1 = tf.reshape(h1, [batch_size, 8, 8, 256])
        h1 = tf.nn.relu(g_bn1(h1))
        h1 = conv_cond_concat(h1, label_conv)

        # deconv layers
        h2 = deconv2d(h1, [batch_size, 16, 16, 256],
                      5, 5, 2, 2, name='g_h2')
        h2 = tf.nn.relu(g_bn2(h2))
        h2 = conv_cond_concat(h2, label_conv)

        h3 = deconv2d(h2, [batch_size, 16, 16, 256],
                      5, 5, 1, 1, name='g_h3')
        h3 = tf.nn.relu(g_bn3(h3))
        h3 = conv_cond_concat(h3, label_conv)

        h4 = deconv2d(h3, [batch_size, 32, 32, 256],
                      5, 5, 2, 2, name='g_h4')
        h4 = tf.nn.relu(g_bn4(h4))
        h4 = conv_cond_concat(h4, label_conv)

        h5 = deconv2d(h4, [batch_size, 32, 32, 256],
                      5, 5, 1, 1, name='g_h5')
        h5 = tf.nn.relu(g_bn5(h5))
        h5 = conv_cond_concat(h5, label_conv)

        h6 = deconv2d(h5, [batch_size, 64, 64, 128],
                      5, 5, 2, 2, name='g_h6')
        h6 = tf.nn.relu(g_bn6(h6))
        h6 = conv_cond_concat(h6, label_conv)

        h7 = deconv2d(h6, [batch_size, 128, 128, 64],
                      5, 5, 2, 2, name='g_h7')
        h7 = tf.nn.relu(g_bn7(h7))
        h7 = conv_cond_concat(h7, label_conv)

        h8 = deconv2d(h7, [batch_size, 128, 128, 3],
                      5, 5, 1, 1, name='g_h8')
        h8 = tf.nn.tanh(h8)

        return h8
