'''
Created on Sep 27, 2017

@author: Sara
'''
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
import random

class GameSolver(object):

    def __init__(self, utility, learning_rate=1e-3, batch_size=100, input_dim=4, n=2):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n=n
        self.utility = utility
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.input_dim])

        f1 = fc(self.x, 10, scope='fc1', activation_fn=tf.nn.elu)
        f2 = fc(f1, 10, scope='fc2', activation_fn=tf.nn.elu)
        self.strategy = fc(f2, self.n, scope='fc3', activation_fn=tf.nn.sigmoid)
        self.strategy=self.strategy/tf.reduce_sum(self.strategy)
        self.u = self.utility(self.x, self.strategy, self.input_dim)
        # Loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.u)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, utility = self.sess.run(
            [self.train_op, self.u],
            feed_dict={self.x: x}
        )
        return utility
    def solve_game(self, x):
        strategy = self.sess.run(self.strategy, feed_dict={self.x: x})
        return strategy

def utility_fn(game, strategy, dim):
    min = 0
    sminus = 1-strategy
    #s1 = game[:][:dim]*strategy
    #s2 = game[:][dim:]*sminus
    s = tf.concat([-strategy,sminus],1)
    util = tf.reduce_max(game*s,1)
    #c1 = tf.greater(strategy, 1) 
    #c2 = tf.greater(tf.reduce_sum(strategy,1),1)

    #cu1 = tf.reduce_sum(tf.where(c1, 1000*tf.ones_like(strategy), tf.zeros_like(strategy)),1)
    #cu2 = tf.where(c2, 1000*tf.ones_like(util), tf.zeros_like(util))
    #v = util+cu1+cu2
    min = tf.reduce_mean(util)
    return min

def create_data(n, dim=2):
    data = []
    for i in range(n):
        m = [random.random(),random.random(),random.random(),random.random()]
        data.append(m)
    tf.constant(data)
    return data

def trainer(learning_rate=1e-3, batch_size=1, dim=2):
    num_sample=200
    num_epoch = num_sample/batch_size
    
    data = create_data(num_sample)
    
    model = GameSolver(utility_fn, learning_rate=learning_rate,
                                    batch_size=batch_size, n=dim)

    for epoch in range(num_epoch):
        for iter in range(num_sample // batch_size):
            # Obtina a batch
            batch = data[epoch*batch_size:(epoch+1)*batch_size]
            # Execute the forward and the backward pass and report computed losses
            utility = model.run_single_step(batch)

        if epoch % 5 == 0:
            print('[Epoch {}] Utility: {}'.format(
                epoch, utility))

    print('Done!')
    return model

# Train the model
model = trainer(learning_rate=1e-4,  batch_size=1, dim=2)
s = model.solve_game([[0,0,0.5,1]])
print s
print utility_fn([0,0,0.5,1], s, 2).eval()