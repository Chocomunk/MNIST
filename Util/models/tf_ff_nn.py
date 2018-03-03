import tensorflow as tf
import numpy as np


class NeuralNetwork:
    """Represents a Neural Network
    
    Attributes:
        biases (list of tensorflow Tensor)          List of bias values for each connection in the network
        weights (list of tensorflow Tensor)         List of weights for each connection in the network
        x (tensorflow Tensor)                   Input layer of the network. Represented as a placeholder
                                                        to allow the computation to be executed in a tensorflow
                                                        session
        y (tensorflow Tensor)              Training output data. Represented as a placeholder to
                                                        allow the training to occur in a tensorflow session
        layer_values (list of tensorflow Tensor)    List of tensors representing the nodes of every layer
        cross_entropy (tensorflow Tensor)               Representation of the cross-entropy of the cost of the
                                                        approximated netwokr output and training output
        train_step (tensorflow Operation)           Represents the training operation executed on the calculated
                                                        cross-entropy
    """

    def __init__(self, network, cost_func, activation_func, train_func, write_summaries=True):
        """Creates a NeuralNetwork object from a simple specified topology
        
        Args:
            network (list of int):                  List of integers representing the number of nodes of each layer
                                                        - Must have at least 2 layers for input and output
            cost_func (function)                    Method of cross-entropy cost to use in this network
            activation_func (function)              Activation function to compute for the output of every node
            train_func (function)                   Training function to use to train the model
        """
        self.write_summaries = write_summaries
        self.biases = []
        self.weights = []
        self.x = tf.placeholder(tf.float32, shape=[None, network[0]], name='input')
        self.y = tf.placeholder(tf.float32, shape=[None, network[len(network) - 1]], name='training_output')
        self.layer_values = [self.x]

        # Setup network
        for i in range(1, len(network)):
            layer_name = 'Layer_{}'.format(i)
            with tf.name_scope(layer_name):
                with tf.name_scope('weights'):
                    weight = tf.Variable(
                        tf.random_normal([network[i - 1], network[i]], stddev=0.05, name=layer_name + '/weights'))
                    self.weights.append(weight)
                    self.variable_summaries(weight)
                with tf.name_scope('biases'):
                    bias = tf.Variable(tf.constant(0.1, shape=[network[i]], name=layer_name + '/biases'))
                    self.biases.append(bias)
                    self.variable_summaries(bias)
                with tf.name_scope('activation'):
                    activation = activation_func(tf.add(
                        tf.matmul(self.layer_values[i - 1], self.weights[i - 1]), self.biases[i - 1],
                        name=layer_name + '/preactivation'),
                        name='activation')
                    self.layer_values.append(activation)
                    tf.summary.histogram(layer_name + '/activations', activation)

        # Setup training steps
        with tf.name_scope('cross_entropy'):
            cost = cost_func(logits=self.get_output_layer(), labels=self.y)
            with tf.name_scope('total'):
                self.cross_entropy = tf.reduce_mean(cost)
            tf.summary.scalar('cross entropy', self.cross_entropy)
        with tf.name_scope('train'):
            self.train_step = train_func(self.cross_entropy)

    def train(self):
        """Returns training operations to run in a tensorflow session
        
        Returns:
            List of this network's cross-entropy and training operation
        """
        return [self.cross_entropy, self.train_step]

    def get_accuracy(self):
        """Returns the accuracy calculation to run in a tensorflow session
        
        Returns:
            Tensor representing the accuracy measurement of the network
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(self.get_output_layer(), 1),
                                              tf.argmax(self.y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
        return [accuracy]

    def get_feed_dict(self, input_training, output_data):
        """Creates a feed dictionary from given input and output data
        
        Returns:
            The feed dictionary the binds the training placeholders
                of this network with given input and output data
        """
        return {self.x: input_training, self.y: output_data}

    def get_output_layer(self):
        """Returns the output layer of this network
        
        Returns:
            The last element of the layer_values list
        """
        return self.layer_values[len(self.layer_values) - 1]

    def get_train_data(self):
        """Returns the training data of this network
        
        Returns:
            A tuple representing the input of output training data"""
        return self.x, self.y

    def variable_summaries(self, var):
        """Creates summaries for a Tensor"""
        if self.write_summaries:
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

                tf.summary.scalar('mean', mean)
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)
