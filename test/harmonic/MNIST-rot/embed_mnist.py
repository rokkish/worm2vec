"""Embed MNIST-rot"""

import argparse
import os
import random
import sys
import time
#import urllib2
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf
import pandas as pd

from mnist_model import deep_mnist
from run_mnist import download2FileAndExtract, settings, add_folder, minibatcher, get_learning_rate
from tensorboard.plugins import projector
from sprite_images import create_sprite_image, save_sprite_image

import get_logger
logger = get_logger.get_logger(name='embed')


def main(args):
   """The magic happens here"""
   tf.reset_default_graph()
   ##### SETUP AND LOAD DATA #####
   args, data = settings(args)
   logger.debug(args)

   ##### BUILD MODEL #####
   ## Placeholders
   x = tf.placeholder(tf.float32, [args.batch_size,784], name='x')
   y = tf.placeholder(tf.int64, [args.batch_size], name='y')
   learning_rate = tf.placeholder(tf.float32, name='learning_rate')
   train_phase = tf.placeholder(tf.bool, name='train_phase')

   # Construct model and optimizer
   pred = deep_mnist(args, x, train_phase)
   loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y))

   # Evaluation criteria
   correct_pred = tf.equal(tf.argmax(pred, 1), y)
   accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

   # Optimizer
   optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
   grads_and_vars = optim.compute_gradients(loss)
   modified_gvs = []
   # We precondition the phases, for faster descent, in the same way as biases
   for g, v in grads_and_vars:
      if 'psi' in v.name:
         g = args.phase_preconditioner*g
      modified_gvs.append((g, v))
   train_op = optim.apply_gradients(modified_gvs)

   ##### TRAIN ####
   # Configure tensorflow session
   init_global = tf.global_variables_initializer()
   init_local = tf.local_variables_initializer()
   config = tf.ConfigProto()
   config.gpu_options.allow_growth = False
   config.gpu_options.visible_device_list = "0"
   config.log_device_placement = True

   lr = args.learning_rate
   saver = tf.train.Saver()
   sess = tf.Session(config=config)
   sess.run([init_global, init_local], feed_dict={train_phase : True})

   saver.restore(sess, args.checkpoint_path)
   
   start = time.time()
   epoch = 0
   step = 0.
   counter = 0
   best = 0.

   # TEST
   """
   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   test_acc = 0.
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      test_acc += accuracy_
      sys.stdout.write('Testing\r')
      sys.stdout.flush()
   test_acc /= (i+1.)
   
   logger.info('Test Acc.: {:04f}'.format(test_acc))
   
   """
   # Embed data
   accuracy_summary = tf.summary.scalar("ACC", accuracy)

   # config of Embed
   import datetime
   str_date = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
   logdir = "./logs/tensorboard/{}/".format(str_date)
   os.makedirs(logdir, exist_ok=True)
   #embed_dir = logdir + "/embedding-ckpt"
   #os.makedirs(embed_dir, exist_ok=True)

   # access weight
   layers_name = [
      "block1/hconv1/Reshape_1:0", 
      "block2/hconv3/Reshape_1:0", 
      "block3/hconv5/Reshape_1:0", 
      "block4/hconv7/Reshape_1:0", 
      "block4/hconv7/Reshape:0", 
      "block4/Mean:0",
      "block4/Maximum:0"
   ]
   medium_op = list(map(lambda tname:tf.get_default_graph().get_tensor_by_name(tname), layers_name))


   batcher = minibatcher(data['test_x'], data['test_y'], args.batch_size)
   for i, (X, Y) in enumerate(batcher):
      feed_dict = {x: X, y: Y, train_phase: False}
      summary = sess.run(medium_op, feed_dict=feed_dict)

      # select last layer output
      summary_np = np.array(summary[-1])
      summary_np = np.reshape(summary_np, (summary_np.shape[0], -1))
      variables = tf.Variable(summary_np, trainable=False, name="embedding_lastlayer")

      # make tensor.tsv
      df = pd.DataFrame(summary_np).astype("float64")
      df.to_csv(logdir + "tensor.csv", header=False, index=False, sep="\t")

      # make metadata.tsv (labels)
      with open(logdir + "metadata.tsv", "w") as f:
         f.write("Index\tLabel\n")
         for index, label in enumerate(Y):
            f.write("%d\t%d\n" % (index, int(label)))

      # make sprite image (labels)
      # X = (46, 784)
      save_sprite_image(create_sprite_image(X), path=logdir + "sprite.png")
      break

   #saver.save(sess, embed_dir)

   # config of projector
   config_projector = projector.ProjectorConfig()
   embedding = config_projector.embeddings.add()
   embedding.tensor_name = variables.name
   # tsv path
   embedding.tensor_path = "tensor.csv"
   embedding.metadata_path = "metadata.tsv"
   # config of sprite image
   embedding.sprite.image_path = "sprite.png"
   embedding.sprite.single_image_dim.extend([28, 28])

   summary_writer = tf.summary.FileWriter(logdir, sess.graph)
   projector.visualize_embeddings(summary_writer, config_projector)

   sess.close()


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--data_dir", help="data directory", default='./data')
   parser.add_argument("--default_settings", help="use default settings", type=bool, default=True)
   parser.add_argument("--combine_train_val", help="combine the training and validation sets for testing", type=bool, default=False)
   main(parser.parse_args())
