import tensorflow as tf
import numpy as np

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, 
                 action_size=2, hidden_size=10, step_size=1 ,
                 name='QNetwork'):    
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None,step_size, state_size], name='inputs_')
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
        
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_size)
            self.lstm_out, self.state = tf.nn.dynamic_rnn(self.lstm,self.inputs_,dtype=tf.float32)
            
            self.reduced_out = self.lstm_out[:,-1,:]
            self.reduced_out = tf.reshape(self.reduced_out,shape=[-1,hidden_size])

            self.w2 = tf.Variable(tf.random_uniform([hidden_size,hidden_size]))
            self.b2 = tf.Variable(tf.constant(0.1,shape=[hidden_size]))
            self.h2 = tf.matmul(self.reduced_out,self.w2) + self.b2
            self.h2 = tf.nn.relu(self.h2)
            self.h2 = tf.contrib.layers.layer_norm(self.h2)
            
            self.w3 = tf.Variable(tf.random_uniform([hidden_size,action_size]))
            self.b3 = tf.Variable(tf.constant(0.1,shape=[action_size]))
            self.output = tf.matmul(self.h2,self.w3) + self.b3

            #self.output = tf.contrib.layers.layer_norm(self.output)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
