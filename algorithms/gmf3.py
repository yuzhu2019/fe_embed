## Yu Zhu, Rice ECE, yz126@rice.edu
## reference: https://colab.research.google.com/drive/1kDnPNoNjW8weHtqEtta6AcoYJsambZ0Y
## tensorflow version 2

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

class MFEmbedder(Model):
    def __init__(self, init_func, vsize, wsize, embdim):
        super(MFEmbedder, self).__init__()
        self.V = tf.Variable(init_func(vsize, embdim), name="vectors")
        self.W = tf.Variable(init_func(wsize, embdim), name="covectors")

    def call(self, x=None):
        return tf.matmul(self.V, self.W, transpose_b=True)

class SGNSLoss:
    def __init__(self, Nij_p, Nij_n):  
        self.Nij_p = Nij_p
        self.Nij_n = Nij_n

    def __call__(self, M_hat):
        pos = self.Nij_p * tf.math.log_sigmoid(M_hat)
        neg = self.Nij_n * tf.math.log_sigmoid(-M_hat)
        obj = tf.reduce_sum(pos) + tf.reduce_sum(neg) 
        return -obj
    
def make_train_step(loss_obj):
    optimizer = tf.keras.optimizers.Adam(lr=0.1) # optimizer and learning rate 
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    @tf.function
    def train_step(scope_model):
        with tf.GradientTape() as tape:
            mhat = scope_model(None) 
            loss = tf.reduce_sum(loss_obj(mhat))
        gradients = tape.gradient(loss, scope_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, scope_model.trainable_variables))
        train_loss(loss)

    return train_loss, train_step

def GMF(Nij_p_np, Nij_n_np, embed_dim, n_iters, plot):
    graph_size = np.size(Nij_p_np, 0)
    feature_size = np.size(Nij_p_np, 1)
    Nij_p = tf.convert_to_tensor(Nij_p_np, dtype=tf.float32)
    Nij_n = tf.convert_to_tensor(Nij_n_np, dtype=tf.float32)
    normal_weights_initialization = lambda v,e: tf.random.normal((v,e), 0.0, 1.0/e)
    model = MFEmbedder(normal_weights_initialization, graph_size, feature_size, embed_dim)
    loss_obj = SGNSLoss(Nij_p=Nij_p, Nij_n=Nij_n)
    train_loss, train_step = make_train_step(loss_obj)
    results = []
    for i in range(n_iters): 
        train_step(model) 
        results.append(train_loss.result())
        
    if plot:
        x = np.arange(len(results))
        y = np.array(results)

        _ = plt.figure()
        _ = plt.plot(x, y, '--r', label="loss")
        _ = plt.xlabel("Iteration")
        _ = plt.ylabel("Loss value")
        _ = plt.legend()
        plt.show()
    
    return model

