import tensorflow as tf
import numpy as np

class LSTMLayer(tf.keras.Layer):
    def __init__(self, output_dim, input_dim, trans1="tanh", trans2="tanh"):
        '''
        Args:
            input_dim (int):       dimensionality of input data
            output_dim (int):      number of outputs for LSTM layers
            trans1, trans2 (str):  activation functions used inside the layer;
                                   one of: "tanh" (default), "relu" or "sigmoid"

        Returns: customized Keras layer object used as intermediate layers in DGM
        '''
        super(LSTMLayer, self).__init__()

        # Add properties for layer including activation functions
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Activation function mapping
        activation_map = {
            "tanh": tf.nn.tanh,
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid
        }

        self.trans1 = activation_map.get(trans1, tf.nn.tanh)
        self.trans2 = activation_map.get(trans2, tf.nn.tanh)

        # LSTM layer parameters (Xavier initialization)
        # u vectors (weighting vectors for inputs original inputs x)
        self.Uz = self.add_weight(name="Uz", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Ug = self.add_weight(name="Ug", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Ur = self.add_weight(name="Ur", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Uh = self.add_weight(name="Uh", shape=(self.input_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())

        # w vectors (weighting vectors for output of previous layer)
        self.Wz = self.add_weight(name="Wz", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Wg = self.add_weight(name="Wg", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Wr = self.add_weight(name="Wr", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())
        self.Wh = self.add_weight(name="Wh", shape=(self.output_dim, self.output_dim),
                                  initializer=tf.keras.initializers.GlorotNormal())

        # bias vectors
        self.bz = self.add_weight(name="bz", shape=[1, self.output_dim])
        self.bg = self.add_weight(name="bg", shape=[1, self.output_dim])
        self.br = self.add_weight(name="br", shape=[1, self.output_dim])
        self.bh = self.add_weight(name="bh", shape=[1, self.output_dim])

    def call(self, S, X):
        '''Compute output of a LSTMLayer for a given inputs S,X.

        Args:
            S: output of previous layer
            X: data input

        Returns: customized Keras layer object used as intermediate layers in DGM
        '''
        # Compute components of LSTM layer output
        Z = self.trans1(tf.add(tf.add(tf.matmul(X, self.Uz), tf.matmul(S, self.Wz)), self.bz))
        G = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ug), tf.matmul(S, self.Wg)), self.bg))
        R = self.trans1(tf.add(tf.add(tf.matmul(X, self.Ur), tf.matmul(S, self.Wr)), self.br))

        H = self.trans2(tf.add(tf.add(tf.matmul(X, self.Uh), tf.matmul(tf.multiply(S, R), self.Wh)), self.bh))

        # Compute LSTM layer output
        S_new = tf.add(tf.multiply(tf.subtract(tf.ones_like(G), G), H), tf.multiply(Z, S))

        return S_new


class DenseLayer(tf.keras.Layer):
    def __init__(self, output_dim, input_dim, transformation=None):
        '''
        Args:
            input_dim:       dimensionality of input data
            output_dim:      number of outputs for dense layer
            transformation:  activation function used inside the layer; using
                             None is equivalent to the identity map

        Returns: customized Keras (fully connected) layer object
        '''
        super(DenseLayer, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        # Dense layer parameters (use Xavier initialization)
        self.W = self.add_weight(name= "W", shape=(self.input_dim, self.output_dim),
                                 initializer=tf.keras.initializers.GlorotNormal())

        # Bias vectors
        self.b = self.add_weight(name="b", shape=(1, self.output_dim))

        # Set transformation function
        if transformation:
            activation_map = {
                "tanh": tf.tanh,
                "relu": tf.nn.relu
            }
            self.transformation = activation_map.get(transformation, None)
        else:
            self.transformation = None

    def call(self, X):
        '''Compute output of a dense layer for a given input X

        Args:
            X: input to layer
        '''
        # Compute dense layer output
        S = tf.add(tf.matmul(X, self.W), self.b)

        if self.transformation:
            S = self.transformation(S)

        return S


class DGMNet(tf.keras.Model):
    def __init__(self, layer_width, n_layers, input_dim, final_trans=None):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer

        Returns: customized Keras model object representing DGM neural network
        '''
        super(DGMNet, self).__init__()

        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = [
            LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
        ]

        # Final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(1, layer_width, transformation=final_trans)

    def call(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs
        X = tf.concat([t,x],1)

        # Call initial layer
        S = self.initial_layer.call(X)

        # Call intermediate LSTM layers
        for lstm_layer in self.LSTMLayerList:
            S = lstm_layer.call(S, X)

        # Call final layer
        result = self.final_layer.call(S)

        return result

class PIANet(tf.keras.Model):
    def __init__(self, layer_width, n_layers, input_dim, num_classes, final_trans=None):
        '''
        Args:
            layer_width:
            n_layers:    number of intermediate LSTM layers
            input_dim:   spatial dimension of input data (EXCLUDES time dimension)
            final_trans: transformation used in final layer

        Returns: customized Keras model object representing DGM neural network
        '''
        super(PIANet, self).__init__()

        # Initial layer as fully connected
        # NOTE: to account for time inputs we use input_dim+1 as the input dimensionality
        self.initial_layer = DenseLayer(layer_width, input_dim+1, transformation="tanh")

        # Intermediate LSTM layers
        self.n_layers = n_layers
        self.LSTMLayerList = [
            LSTMLayer(layer_width, input_dim+1) for _ in range(self.n_layers)
        ]

        # Final layer as fully connected with a single output (function value)
        self.final_layer = DenseLayer(num_classes, layer_width, transformation=final_trans)

    def call(self, t, x):
        '''
        Args:
            t: sampled time inputs
            x: sampled space inputs

        Run the DGM model and obtain fitted function value at the inputs (t,x)
        '''
        # Define input vector as time-space pairs
        X = tf.concat([t, x], 1)

        # Call initial layer
        S = self.initial_layer.call(X)

        # Call intermediate LSTM layers
        for lstm_layer in self.LSTMLayerList:
            S = lstm_layer.call(S, X)

        # Call final layer
        result = self.final_layer.call(S)
        #result = tf.nn.softmax(result)
        op = tf.math.argmax(result, 1)
        op = tf.reshape(op, [tf.shape(op)[0],1]).astype('float32')
        return op, result