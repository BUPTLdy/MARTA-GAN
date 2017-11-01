import os
import sys
import scipy.misc
import pprint
import numpy as np
import time
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from glob import glob
from random import shuffle
from utils import *
from network import *
pp = pprint.PrettyPrinter()

"""
TensorLayer implementation of DCGAN to generate face image.

Usage : see README.md
"""

flags = tf.app.flags
flags.DEFINE_integer("epoch", 100, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The number of batch images [64]")
flags.DEFINE_integer("image_size", 256, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_integer("output_size", 256, "The size of the output images to produce [64]")
flags.DEFINE_integer("sample_size", 64, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("sample_step", 500, "The interval of generating sample. [500]")
flags.DEFINE_integer("save_step", 50, "The interval of saveing checkpoints. [500]")
flags.DEFINE_string("dataset", "uc_train_256_data", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS






def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    z_dim = 100

    # with tf.device("/gpu:0"): # <-- if you have a GPU machine
    z = tf.placeholder(tf.float32, [FLAGS.batch_size, z_dim], name='z_noise')
    real_images =  tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.output_size, FLAGS.output_size, FLAGS.c_dim], name='real_images')

    # z --> generator for training
    net_g, g_logits = generator_simplified_api(z, is_train=True, reuse=False)
    # generated fake images --> discriminator
    net_d, d_logits, feature_fake = discriminator_simplified_api(net_g.outputs, is_train=True, reuse=False)
    # real images --> discriminator
    net_d2, d2_logits, feature_real = discriminator_simplified_api(real_images, is_train=True, reuse=True)
    # sample_z --> generator for evaluation, set is_train to False
    # so that BatchNormLayer behave differently
    net_g2, g2_logits = generator_simplified_api(z, is_train=False, reuse=True)

    #
    net_d3, d3_logits, _ = discriminator_simplified_api(real_images, is_train=False, reuse=True)

    # cost for updating discriminator and generator
    # discriminator: real images are labelled as 1
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d2_logits, labels=tf.ones_like(d2_logits)))    # real == 1
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.zeros_like(d_logits)))     # fake == 0
    d_loss = d_loss_real + d_loss_fake
    # generator: try to make the the fake images look real (1)
    g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits, labels=tf.ones_like(d_logits)))
    g_loss2 = tf.reduce_mean(tf.nn.l2_loss(feature_real-feature_fake))/(FLAGS.image_size*FLAGS.image_size)
    g_loss = g_loss1+g_loss2
    #g_loss = tf.reduce_mean(tf.abs(feature_real-feature_fake))
    # trainable parameters for updating discriminator and generator
    g_vars = net_g.all_params   # only updates the generator
    d_vars = net_d.all_params   # only updates the discriminator

    net_g.print_params(False)
    print("---------------")
    net_d.print_params(False)

    # optimizers for updating discriminator and generator
    d_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=FLAGS.beta1) \
                      .minimize(g_loss, var_list=g_vars)

    sess=tf.Session()
    tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.88)
    sess.run(tf.initialize_all_variables())

    # load checkpoints
    print("[*] Loading checkpoints...")
    model_dir = "%s_%s_%s" % (FLAGS.dataset, 64, FLAGS.output_size)
    save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
    # load the latest checkpoints
    #for num in xrange(70, 71):
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    print net_g_name, net_d_name

    if not (os.path.exists(net_g_name) and os.path.exists(net_d_name)):
        print("[!] Loading checkpoints failed!")
    else:
        net_g_loaded_params = tl.files.load_npz(name=net_g_name)
        net_d_loaded_params = tl.files.load_npz(name=net_d_name)
        tl.files.assign_params(sess, net_g_loaded_params, net_g)
        tl.files.assign_params(sess, net_d_loaded_params, net_d)
        print("[*] Loading checkpoints SUCCESS!")


    # TODO: use minbatch to shuffle and iterate
    data_files = glob(os.path.join("./data", FLAGS.dataset, "*.jpg"))


    # TODO: shuffle sample_files each epoch
    sample_seed = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
    if FLAGS.is_train:

        iter_counter = 0
        for epoch in range(FLAGS.epoch):
            #shuffle data
            shuffle(data_files)
            print("[*]Dataset shuffled!")

            # update sample files based on shuffled data
            sample_files = data_files[0:FLAGS.batch_size]
            sample = [get_image(sample_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for sample_file in sample_files]
            sample_images = np.array(sample).astype(np.float32)
            print sample_images.shape
            print("[*]Sample images updated!")

            # load image data
            batch_idxs = min(len(data_files), FLAGS.train_size) // FLAGS.batch_size

            for idx in xrange(batch_idxs):
                batch_files = data_files[idx*FLAGS.batch_size:(idx+1)*FLAGS.batch_size]
                # get real images
                batch = [get_image(batch_file, FLAGS.image_size, is_crop=FLAGS.is_crop, resize_w=FLAGS.output_size, is_grayscale = 0) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_z = np.random.uniform(low=-1, high=1, size=(FLAGS.batch_size, z_dim)).astype(np.float32)
                start_time = time.time()
                # updates the discriminator
                errD, _ = sess.run([d_loss, d_optim], feed_dict={z: batch_z, real_images: batch_images })
                # updates the generator, run generator twice to make sure that d_loss does not go to zero (difference from paper)
                for _ in range(2):
                    errG, _ = sess.run([g_loss, g_optim], feed_dict={z: batch_z, real_images: batch_images})
                print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, FLAGS.epoch, idx, batch_idxs,
                            time.time() - start_time, errD, errG))
                sys.stdout.flush()

                iter_counter += 1
            if np.mod(epoch, 1) == 0:
                # generate and visualize generated images
                #img, errD, errG = sess.run([net_g2.outputs, d_loss, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                img, errG = sess.run([net_g2.outputs, g_loss], feed_dict={z : sample_seed, real_images: sample_images})
                D, D_, errD = sess.run([net_d3.all_layers, net_d3.outputs, d_loss_real], feed_dict={real_images: sample_images})

                '''
                img255 = (np.array(img) + 1) / 2 * 255
                tl.visualize.images2d(images=img255, second=0, saveable=True,
                                name='./{}/train_{:02d}_{:04d}'.format(FLAGS.sample_dir, epoch, idx), dtype=None, fig_idx=2838)
                '''
                save_images(img, [8, 8],
                            './{}/train_{:02d}.png'.format(FLAGS.sample_dir, epoch))
                print("[Sample] d_loss: %.8f, g_loss: %.8f" % (errD, errG))
#                for i in range(len(D)):
#                    print D[i].shape
                #print D[-1], D_, sigmoid(D[-1]), sigmoid(D[-1])==D_
                sys.stdout.flush()

            if np.mod(epoch, 5) == 0:
                print epoch
                # save current network parameters
                print("[*] Saving checkpoints...")
                model_dir = "%s_%s_%s" % (FLAGS.dataset, FLAGS.batch_size, FLAGS.output_size)
                save_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # the latest version location
                net_g_name = os.path.join(save_dir, str(epoch)+'net_g.npz')
                net_d_name = os.path.join(save_dir, str(epoch)+'net_d.npz')
                # this version is for future re-check and visualization analysis
#                    net_g_iter_name = os.path.join(save_dir, 'net_g_%d.npz' % iter_counter)
#                    net_d_iter_name = os.path.join(save_dir, 'net_d_%d.npz' % iter_counter)
                tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
                tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
#                    tl.files.save_npz(net_g.all_params, name=net_g_iter_name, sess=sess)
#                    tl.files.save_npz(net_d.all_params, name=net_d_iter_name, sess=sess)
                print("[*] Saving checkpoints SUCCESS!")



if __name__ == '__main__':
    tf.app.run()
