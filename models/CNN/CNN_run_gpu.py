from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf
import data_processor
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import *
import pickle
from collections import Counter
from tensorflow.contrib import learn
import data_processor as dp
from nltk import word_tokenize
import time
import CNN_run_gpu
from cnn_run_gpu import CNN
import os
from sklearn.metrics import f1_score



train_x=np.load("../../Objects/ShortData/x_train_sum2K.npy", allow_pickle=True)
dev_x=np.load("../../Objects/ShortData/x_dev_sum2K500.npy",allow_pickle=True)

train_y=np.load("../../Objects/ShortData/y_train_sum2K.npy", allow_pickle=True)
dev_y =np.load("../../Objects/ShortData/y_dev_sum2K500.npy", allow_pickle=True)

logfile="/home/nyao111/266/FinalProject/MIDS-W266-Final-MSY/CNN_GPU/logs/CNNS_noH_trial19s.log"

vocabsize=np.max( np.vstack((x_train ,x_dev)))+1
print "Train Shape: ",  x_train.shape
print "Dev Shape: ", x_dev.shape

with open(logfile, "w") as log:
    now = time.strftime("%c")
    log.write("Start time: %s\n" %now)
    log.write("Put everything in cnn_net to GPU\n")
    log.write("Vocab Size: {}\n".format(vocabsize))

with open(logfile, "a") as log:
    log.write("X Train Shape: "+str(x_train.shape)+"\n")
    log.write("X Dev Shape: "+str(x_dev.shape)+"\n")
# For use with GPU

# Hyperparameters
######################################################
M = 150  #Embedding Size
FS = [20, 25, 30] #Filter Sizes
Num_F = 400  #Number of filters per filte rsize
dropout_keep_prob = 0.5
L = 50
batch_size = 75
num_epoch = 50

print "Embedding size : ", M
print "Filter size : ", FS
print "Number of Filters :", Num_F
print "Dropout keep probability : ", dropout_keep_prob
print "Regularization : ", L
print "Batch size : ", batch_size
print "Epoch Number : ", num_epoch

with open(logfile, "a") as log:
    log.write("Embedding size: %d\n" %M )
    log.write("Filter size : %s \n" %(str(FS)))
    log.write("Number of Filters : %d\n" %Num_F)
    log.write("Dropout keep probability : %3.2f\n" %dropout_keep_prob)
    log.write("Regularization : %3.1f \n" %L)
    log.write("Batch size : %d \n" %batch_size)
    log.write("Epoch Number : %d \n" %num_epoch)
#############################    Training graph   #####################
# Get a list of GPU devices
devicelist=['/job:localhost/replica:0/task:0/gpu:0', '/job:localhost/replica:0/task:0/gpu:1','/job:localhost/replica:0/task:0/gpu:2','/job:localhost/replica:0/task:0/gpu:3', '/job:localhost/replica:0/task:0/gpu:4', '/job:localhost/replica:0/task:0/gpu:5' ]
with tf.Graph().as_default():
    for device in devicelist:
        sess = tf.Session(config=tf.ConfigProto(device_filters=device, log_device_placement=True))
        with sess.as_default():
            cnn = CNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocabsize,
                embedding_size = M,
                filter_sizes = FS,
                num_filters = Num_F,
                l2_reg_lambda = L)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Output directory for checkpoints and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            with open(logfile, "a") as log:
                log.write("Writing to {}\n".format(out_dir))


            # Train Summaries

            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  #cnn.input_suprise: s_batch,
                  cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step,loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 500 == 0:
                  print("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                  with open(logfile, "a") as log:
                      log.write("{}: step {}, loss {:g}, acc {:g} \n".format(time_str, step, loss, accuracy))


            # Development data can cause memory problems, therefore, it is divided into smaller portions
            def dev_step(devbatches, writer=None):
                """
                Evaluates model on a dev set
                """
                predicts=[]
                actuals=[]

                for batch in devbatches:
                    x_batch_dev, y_batch_dev=zip(*batch)
                    feed_dict = {
                      cnn.input_x: x_batch_dev,
                      cnn.input_y: y_batch_dev,
                      #cnn.input_suprise: s_batch,
                      cnn.dropout_keep_prob: dropout_keep_prob
                    }
                    step, loss, prediction, label = sess.run(
                        [global_step, cnn.loss, cnn.predictions, cnn.labels],
                        feed_dict)
                    predicts.append(prediction)
                    actuals.append(label)

                pred=np.concatenate(predicts)
                y_dev=np.concatenate(actuals)
                f1=f1_score(y_dev, pred, average='weighted')
                accuracy=np.sum(pred==y_dev)/len(pred)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, dev acc {}, f1-score {}".format(time_str, step, loss, accuracy, f1))
                with open(logfile, "a") as log:
                    log.write("{}: step {}, loss {:g}, dev acc {:g},f1-score {:g} \n".format(time_str, step, loss, accuracy, f1))
                    log.write("\n")
                    log.write("Training Steps: \n")

            # Generating batches
            batches = data_processor.batch_iter(
                list(zip(train_x, train_y)), batch_size, num_epoch)

            # Training executions
            devbatches=data_processor.batch_iter(list(zip(dev_x, dev_y)), 100, 1)
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % 6000 == 0:
                    with open(logfile, "a") as log:
                        log.write("\nDev Set Evaluation:\n")

                    print("\nDev Set Evaluation:")
                    dev_step(devbatches)
