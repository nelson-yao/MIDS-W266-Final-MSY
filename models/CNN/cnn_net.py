import tensorflow as tf
import numpy as np


class CNN(object):
    """
    Convolution Neural Net with word embedding for Text Classification
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_suprise = tf.placeholder(tf.float32, [None, 1], name="input_surprise")

        # L2 loss function
        l2_loss = tf.constant(0.0)

        # Embedding layer
        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Convolution + maxpool layer give each filter
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],        #strides
                    padding="VALID",
                    name="conv")
                
                # Relu
                relu = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                
                #  Standard maxpool
                pooled = tf.nn.max_pool(
                    relu,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        num_filters_total = num_filters * len(filter_sizes) + 1
        self.h_pool = tf.concat(3, pooled_outputs)
        #features = tf.concat(1, [self.input_suprise, self.h_pool])
        #Concatnate  earning suprise with  word embedding
        self.h_pool_flat = tf.concat(1, [tf.reshape(self.h_pool, [-1, num_filters_total -1]), self.input_suprise])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Initial prediction
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.xw_out = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.xw_out, 1, name="predictions")

        # Loss Function
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.xw_out, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            
        # Calc Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
