import os
import h5py
import socket
import struct
import pickle
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import pickle

def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

# Função para plotar imagens geradas
def plot_generated_images(epoch, client_id, generator, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, size=[examples, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'gan_generated_image_round_{epoch}_client_{client_id}.png')


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


client_order = int(input("client_order(start from 0): "))

# Load a subset of MNIST to simulate the local data partition
(x_train, y_train), (x_test, y_test) = load_partition(client_order)

# Normaliza as imagens para o intervalo [-1, 1]
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train.reshape(x_train.shape[0], 784)

# Tamanho do vetor de entrada para o gerador
latent_dim = 100

# Inicializador de pesos para as camadas da GAN
initializer = initializers.RandomNormal(mean=0.0, stddev=0.02)

# Constrói o gerador
generator = Sequential()
generator.add(Dense(256, input_dim=latent_dim, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(512, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(1024, kernel_initializer=initializer))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Dense(784, activation='tanh', kernel_initializer=initializer))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Constrói o discriminador
discriminator = Sequential()
discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(512, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(256, kernel_initializer=initializer))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Constrói a GAN combinando o gerador e o discriminador
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
x = generator(gan_input)
gan_output = discriminator(x)
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# ### Set host address and port number
host = "127.0.1.1"
port = 10080
max_recv = 100000


# ### Open the client socket
s = socket.socket()
s.connect((host, port))


# ## SET TIMER
start_time = time.time()    # store start time
print("timmer start!")


msg = recv_msg(s)
rounds = msg['rounds'] 
client_id = msg['client_id']
epochs = msg['local_epoch']
send_msg(s, len(x_train))


batch_size = 128
# Treina a GAN
batch_count = x_train.shape[0] // batch_size

for r in range(rounds):
    
    weights = recv_msg(s)
    generator.set_weights(weights)
    
    for e in range(epochs):

        for _ in range(batch_count):
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_images = generator.predict(noise)
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9  # Rótulos suavizados para o treinamento estável

            # Treina o discriminador
            d_loss = discriminator.train_on_batch(X, y_dis)

            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            y_gen = np.ones(batch_size)

            # Treina a GAN (o gerador é treinado via gan)
            g_loss = gan.train_on_batch(noise, y_gen)

        print(f'Época {e+1}/{epochs}, Discriminador Loss: {d_loss}, GAN Loss: {g_loss}')

    
    plot_generated_images(r, client_id, generator)

    msg = generator.get_weights()
    send_msg(s, msg)


print('Finished Training')

end_time = time.time()  #store end time
print("Training Time: {} sec".format(end_time - start_time))