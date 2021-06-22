from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras import backend as K

class Autoencoder:
    """
    Deep Convolutional AE with mirrored encoder and decoder.
    """

# Constructor with parameters for the conv layers:
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_stride, latent_space_dim):
        # Now let's assign these arguments to instance attributes:
        self.input_shape = input_shape # [28, 28, 1] for example 28x28 greyscale image.
        self.conv_filters = conv_filters # Each list element is the # of filters per layer
        self.conv_kernels = conv_kernels # Each list element is the size of the kernel for a layer
        self.conv_stride = conv_stride # Each list elements is the stride for a given layer
        self.latent_space_dim = latent_space_dim # int.

        self.encoder = None
        self.decoder = None
        self.model = None

        # The number of conv layers is length of conv_filters or conv_kernels
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None

        # The build method, called when Autoencoder is instantiated
        self._build()

    def summary(self):
        # The classic TF summary, applied to the encoder model:
        self.encoder.summary()

    def _build(self):
        # Three methods:
        self._build_encoder()
        #self._build_decoder()
        #self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
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
        Adds one layer with keras to a graph x of layers
        conv2D + ReLU + BatchNorm
        """
        layer_number = layer_index + 1

        # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_stride[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
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
        latent_space_dim=2 # 2 neurons as final output.
    )

    autoencoder.summary()