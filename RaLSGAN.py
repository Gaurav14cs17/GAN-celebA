from __future__ import division
import os
import time
import math
from glob import glob
import scipy.io as sio
import tensorflow as tf
import numpy as np
from six.moves import xrange

from model import *
from ops import *
from utils import *


class RaLSGAN(object):
    def __init__(
            self,
            sess,
            input_height=450,
            input_width=450,
            crop=True,
            batch_size=64,
            sample_num=64,
            output_height=128,
            output_width=128,
            y_dim=1,
            z_dim=100,
            gf_dim=64,
            df_dim=64,
            gfc_dim=1024,
            dfc_dim=1024,
            c_dim=3,
            dataset_name='default',
            input_fname_pattern='*.jpg',
            checkpoint_dir=None,
            sample_dir=None):
        """
        Args:
          sess: TensorFlow session
          batch_size: The size of batch. Should be specified before training.
          y_dim: (optional) Dimension of dim for y. [None]
          z_dim: (optional) Dimension of dim for Z. [100]
          gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
          df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
          gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
          dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
          c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir

        self.data = glob(os.path.join(
            "./data", self.dataset_name, self.input_fname_pattern))
        self.data.sort()
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(self.data)

        self.data_y = self.load_labels()

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(
                tf.float32, [self.batch_size, self.y_dim], name='y')
        else:
            self.y = None

        if self.crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.G = generator(self.z, self.y)

        self.D_real, self.D_real_logits = discriminator(
            inputs, self.y, reuse=False)
        self.D_fake, self.D_fake_logits = discriminator(
            self.G, self.y, reuse=True)

        self.sampler = sampler(self.z, self.y)

        """loss function"""
        # d_loss
        self.d_loss_real = tf.reduce_mean(
            tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits) - 1))
        self.d_loss_fake = tf.reduce_mean(
            tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits) + 1))
        self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2

        # g_loss
        self.g_loss = (0.5 * tf.reduce_mean(tf.square(self.D_fake_logits - tf.reduce_mean(self.D_real_logits) - 1)) +
                       0.5 * tf.reduce_mean(tf.square(self.D_real_logits - tf.reduce_mean(self.D_fake_logits) + 1)))

        """data visualization"""
        self.z_sum = histogram_summary("z", self.z)
        self.d_real_sum = histogram_summary("d_real", self.D_real)
        self.d_fake_sum = histogram_summary("d_fake", self.D_fake)
        self.G_sum = image_summary("G", self.G)
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.d_loss,
            var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate,
            beta1=config.beta1) .minimize(
            self.g_loss,
            var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except BaseException:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d_fake_sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = self.data[0:self.sample_num]
        sample = [
            get_image(
                sample_file,
                input_height=self.input_height,
                input_width=self.input_width,
                resize_height=self.output_height,
                resize_width=self.output_width,
                crop=self.crop) for sample_file in sample_files]
        sample_inputs = np.array(sample).astype(np.float32)
        sample_labels = self.data_y[0:self.sample_num]

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            self.data = glob(os.path.join(
                "./data", config.dataset, self.input_fname_pattern))
            self.data.sort()

            seed = 547
            np.random.seed(seed)
            np.random.shuffle(self.data)

            batch_idxs = min(
                len(self.data), config.train_size) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = self.data[idx * config.batch_size:
                                        (idx + 1) * config.batch_size]
                batch = [
                    get_image(
                        batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_labels = self.data_y[idx * config.batch_size:
                                           (idx + 1) * config.batch_size]

                batch_z = np.random.uniform(-1, 1,
                                            [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum], feed_dict={
                    self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to
                # zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum], feed_dict={
                                               self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval(
                    {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                errD_real = self.d_loss_real.eval(
                    {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                errG = self.g_loss.eval(
                    {self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})

                counter += 1
                print(
                    "Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" %
                    (epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                            },
                        )
                        save_images(
                            samples, image_manifold_size(
                                samples.shape[0]), './{}/train_{:02d}_{:04d}.png'.format(
                                    config.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" %
                              (d_loss, g_loss))
                    except BaseException:
                        print("one pic error!...")
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def load_labels(self):
        labels = sio.loadmat('imagelabels.mat')
        labels = labels['labels']
        labels = labels.T

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(labels)

        labels = labels / 51 - 1
        labels = np.asarray(labels)

        return labels

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
