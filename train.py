from autoencoder import VAE
import numpy as np
import os

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150

SPECTROGRAMS_PATH = "Dataset/spectrograms/"


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file in file_names:
            file_path = os.path.join(root, file)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    # cast to numpy format for tensorflow
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # 3000 samples -> (3000, 256, 64, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=[512, 256, 128, 64, 32],
        conv_kernels=[3, 3, 3, 3, 3],
        conv_stride=[2, 2, 2, 2, (2, 1)],
        latent_space_dim=128  # 128 neurons as output dim
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    autoencoder2 = VAE.load("model")
    autoencoder2.summary()