import os
import h5py
import socket
import struct
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from threading import Thread
from threading import Lock
import time
from tqdm import tqdm
import copy

import torch


rounds = 5
local_epoch = 1
users = 2 # number of clients

generator = tf.keras.models.load_model('gan_model_gen')
discriminator = tf.keras.models.load_model('gan_model_dis')


#### variables
clientsoclist = [0]*users

start_time = 0
weight_count = 0

global_weights = copy.deepcopy(generator.get_weights())

datasetsize = [0]*users
weights_list = [0]*users

lock = Lock()

### comunication overhead
total_sendsize_list = []
total_receivesize_list = []

client_sendsize_list = [[] for i in range(users)]
client_receivesize_list = [[] for i in range(users)]

train_sendsize_list = [] 
train_receivesize_list = []

### Socket Functions
def send_msg(sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    l_send = len(msg)
    msg = struct.pack('>I', l_send) + msg
    sock.sendall(msg)
    return l_send

def recv_msg(sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # read the message data
    msg =  recvall(sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen

def recvall(sock, n):
    # helper function to receive n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data


def average_weights(w, datasize):
    """
    Returns the average of the weights.
    """
    
    # Scale each weight by its corresponding data size
    for i, data in enumerate(datasize):
        for j in range(len(w[i])):
            w[i][j] = tf.multiply(w[i][j], float(data))
    
    # Create a deep copy of the first set of weights
    w_avg = copy.deepcopy(w[0])

    # Sum the scaled weights for each layer
    for j in range(len(w_avg)):
        for i in range(1, len(w)):
            w_avg[j] = tf.add(w_avg[j], w[i][j])
        
        # Calculate the average by dividing the sum by the total data size
        w_avg[j] = tf.divide(w_avg[j], float(sum(datasize)))

    return w_avg



#### Receiving user before training

def run_thread(func, num_user):
    global clientsoclist
    global start_time
    
    thrs = []
    for i in range(num_user):
        conn, addr = s.accept()
        print('Conntected with', addr)
        # append client socket on list
        clientsoclist[i] = conn
        args = (i, num_user, conn)
        thread = Thread(target=func, args=args)
        thrs.append(thread)
        thread.start()
    print("timmer start!")
    start_time = time.time()    # store start time
    for thread in thrs:
        thread.join()
    end_time = time.time()  # store end time
    print("TrainingTime: {} sec".format(end_time - start_time))


def receive(userid, num_users, conn): #thread for receive clients
    global weight_count
    
    global datasetsize


    msg = {
        'rounds': rounds,
        'client_id': userid,
        'local_epoch': local_epoch
    }

    datasize = send_msg(conn, msg)    #send epoch
    total_sendsize_list.append(datasize)
    client_sendsize_list[userid].append(datasize)

    train_dataset_size, datasize = recv_msg(conn)    # get total_batch of train dataset
    total_receivesize_list.append(datasize)
    client_receivesize_list[userid].append(datasize)
    
    
    with lock:
        datasetsize[userid] = train_dataset_size
        weight_count += 1
    
    train(userid, train_dataset_size, num_users, conn)


### Treino
def train(userid, train_dataset_size, num_users, client_conn):
    global weights_list
    global global_weights
    global weight_count
    global val_acc
    
    for r in range(rounds):
        with lock:
            if weight_count == num_users:
                for i, conn in enumerate(clientsoclist):
                    datasize = send_msg(conn, global_weights)
                    total_sendsize_list.append(datasize)
                    client_sendsize_list[i].append(datasize)
                    train_sendsize_list.append(datasize)
                    weight_count = 0

        client_weights, datasize = recv_msg(client_conn)
        total_receivesize_list.append(datasize)
        client_receivesize_list[userid].append(datasize)
        train_receivesize_list.append(datasize)

        weights_list[userid] = client_weights
        print("User" + str(userid) + "'s Round " + str(r + 1) +  " is done")
        with lock:
            weight_count += 1
            if weight_count == num_users:
                #average
                global_weights = average_weights(weights_list, datasetsize)
                
        


host = socket.gethostbyname(socket.gethostname())
port = 10080
print(host)

s = socket.socket()
s.bind((host, port))
s.listen(5)




run_thread(receive, users)






end_time = time.time()  # store end time
print("TrainingTime: {} sec".format(end_time - start_time))






# print('val_acc list')
# for acc in val_acc:
#     print(acc)

print('\n\n\n')
print('---total_sendsize_list---')
total_size = 0
for size in total_sendsize_list:
#     print(size)
    total_size += size
print("total_sendsize size: {} bytes".format(total_size))
print('\n')

print('---total_receivesize_list---')
total_size = 0
for size in total_receivesize_list:
#     print(size)
    total_size += size
print("total receive sizes: {} bytes".format(total_size) )
print('\n')

for i in range(users):
    print('---client_sendsize_list(user{})---'.format(i))
    total_size = 0
    for size in client_sendsize_list[i]:
#         print(size)
        total_size += size
    print("total client_sendsizes(user{}): {} bytes".format(i, total_size))
    print('\n')

    print('---client_receivesize_list(user{})---'.format(i))
    total_size = 0
    for size in client_receivesize_list[i]:
#         print(size)
        total_size += size
    print("total client_receive sizes(user{}): {} bytes".format(i, total_size))
    print('\n')

print('---train_sendsize_list---')
total_size = 0
for size in train_sendsize_list:
#     print(size)
    total_size += size
print("total train_sendsizes: {} bytes".format(total_size))
print('\n')

print('---train_receivesize_list---')
total_size = 0
for size in train_receivesize_list:
#     print(size)
    total_size += size
print("total train_receivesizes: {} bytes".format(total_size))
print('\n')
