import argparse
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf

import flwr as fl


# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Define Flower client
class MNISTClient(fl.client.NumPyClient):
    def __init__(self, model, generator, latent_dim, x_train, y_train, x_test, y_test):
        self.model = model
        self.generator = generator
        self.latent_dim = latent_dim
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        return self.generator.get_weights()
    
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        generator_weights = train_gan(self.model, self.generator, self.latent_dim, self.x_train, epochs, batch_size)
    
        # # Train the model using hyperparameters from config
        # history = self.model.fit(
        #     self.x_train,
        #     self.y_train,
        #     batch_size,
        #     epochs,
        #     validation_split=0.1,
        # )

        # Return updated model parameters and results
        parameters_prime = generator_weights
        num_examples_train = len(self.x_train)

        return parameters_prime, num_examples_train

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of MNIST to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    ## Load and compile Keras model
    # model = tf.keras.applications.EfficientNetB0(
    #     input_shape=(32, 32, 3), weights=None, classes=10
    # )
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    # Carrega o modelo GAN treinado

    # Tamanho do vetor de entrada para o gerador
    latent_dim = 100


    with open('gan_model.pkl', 'rb') as file:
        generator, discriminator = pickle.load(file)

    # Constrói a GAN combinando o gerador e o discriminador
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

    # Load a subset of MNIST to simulate the local data partition
    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    if args.toy:
        x_train, y_train = x_train[:10], y_train[:10]
        x_test, y_test = x_test[:10], y_test[:10]

    # Start Flower client
    client = MNISTClient(gan, generator, latent_dim, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (
        x_train[idx * 6000 : (idx + 1) * 6000],
        y_train[idx * 6000 : (idx + 1) * 6000],
    ), (
        x_test[idx * 1000 : (idx + 1) * 1000],
        y_test[idx * 1000 : (idx + 1) * 1000],
    )


# Função para treinar a GAN
def train_gan(gan, generator, latent_dim, x_train, epochs=10, batch_size=128):
    batch_count = x_train.shape[0] // batch_size

    for e in range(epochs):
        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_images = generator.predict(noise)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # Rótulos suavizados para o treinamento estável

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)

            # Treina a GAN (o gerador é treinado via gan)
            g_loss = gan.train_on_batch(noise, y_gen)

        print(f'Época {e+1}/{epochs}, GAN Loss: {g_loss}')
        
        # Salva imagens geradas a cada 10 épocas
        if (e + 1) % 10 == 0:
            plot_generated_images(e, generator, latent_dim)

    # Retorna os pesos do gerador ao final do treinamento
    return generator.get_weights()

# Função para plotar imagens geradas
def plot_generated_images(epoch, generator, latent_dim, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_epoch_{epoch}.png')

if __name__ == "__main__":
    main()
