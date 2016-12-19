import tensorflow as tf
import numpy as np


class CNN(object):
    """

    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # put the embedding layer in GPU
        #devicelist=['/job:localhost/replica:0/task:0/gpu:0', '/job:localhost/replica:0/task:0/gpu:1','/job:localhost/replica:0/task:0/gpu:2','/job:localhost/replica:0/task:0/gpu:3', '/job:localhost/replica:0/task:0/gpu:4', '/job:localhost/replica:0/task:0/gpu:5' ]
        #for device in devicelist[:4]:
        #with tf.device(device), tf.name_scope("embedding"):
        with tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # convolution and max pool filters with different sizes
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filterShape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filterShape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(3, pooled_outputs)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

                # Add dropout
                with tf.name_scope("dropout"):
                    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final logits and predictions
        with tf.name_scope("output"), tf.variable_scope("outputlayer"):
            Wout = tf.get_variable("W_Output", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
            bout = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_Output")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, Wout, bout, name="scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Calculate cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.labels=tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions,self.labels )
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
