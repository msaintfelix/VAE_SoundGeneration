from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, \
    Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

import numpy as np
import os
import pickle

class Autoencoder:
    """
    Deep Convolutional AE with mirrored encoder and decoder.
    """

    # First define the constructor with parameters for the conv layers
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_stride, latent_space_dim):

        # Now let's assign these arguments as attributes
        self.input_shape = input_shape # [28, 28, 1] for example 28x28 greyscale image
        self.conv_filters = conv_filters # Each list element is the # of filters per layer
        self.conv_kernels = conv_kernels # Each list element is the size of the kernel for a layer
        self.conv_stride = conv_stride # Each list elements is the stride for a given layer
        self.latent_space_dim = latent_space_dim # int.

        # more attributes
        self.encoder = None
        self.decoder = None
        self.model = None

        # The number of conv layers is length of conv_filters or conv_kernels
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        # The build method, called when Autoencoder is instantiated
        self._build()

    def summary(self):
        # The classic TF summary, applied to the encoder model
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )

    # save to current directory by default
    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)


    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_stride,
            self.latent_space_dim
            ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        # open file in read and binary mode
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    # private methods
    def _build(self):
        # Three methods:
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input)) # decode the output of the encoder to get the model_output
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # the shape needs to be flattened -> use np.prod
        num_neurons = np.prod(self._shape_before_bottleneck) # e.g [1,2,4] -> 8
        # apply the dense_layer to the graph of layers
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        # go from flat to 3 dimensional array and apply to graph
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        # loop through all conv layers in reverse order and stop at the 1st layer
        # exclude layer 0
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_stride[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1, # because one dimensional greyscale images
            kernel_size=self.conv_kernels[0], # here the last block is like the previous first
            strides=self.conv_stride[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        # The bottleneck is also the encoder output
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Creates all conv blocks in the encoder, creates the graph x of layers"""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """
        Add one layer with keras to a graph x of layers
        conv2D + ReLU + BatchNorm
        """
        layer_number = layer_index + 1

        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_stride[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}" # f-string type of print
        )

        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)

        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)"""

        # First we need to save the shape of data before Flatten,
        # for later use with decoder.
        # [batch size, width, height, #of channels]
        # But we don't need the batch size, so [1:]
        self._shape_before_bottleneck = K.int_shape(x)[1:]

        x = Flatten()(x)
        # The Dense has the dimensionality of the latent space
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x

if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=[32, 64, 64, 64],
        conv_kernels=[3, 3, 3, 3],
        conv_stride=[1, 2, 2, 1],
        latent_space_dim=2 # 2 neurons as final output
    )

    autoencoder.summary()