import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 1.0 # starting value of epsilon
FINAL_EPSILON = 0 # final value of epsilon
REPLAY_SIZE = 100000 # experience replay buffer size
BATCH_SIZE = 64 # size of minibatch

class DQN():
  # DQN Agent
  def __init__(self, env, e=-1,model=0):
    # init experience replay
    self.replay_buffer = deque()
    # init some parameters
    self.time_step = 0
    if e == -1: self.epsilon = INITIAL_EPSILON
    else: self.epsilon = e
    self.state_dim = env.observation_space.n
    self.action_dim = env.action_space.n
    self.saver = tf.compat.v1.train.Saver()
    self.create_Q_network(model)
    self.create_training_method()

    # Init session
    self.session = tf.compat.v1.InteractiveSession()
    self.session.run(tf.compat.v1.global_variables_initializer())
    # self.saver = tf.compat.v1.train.Saver()

  def create_Q_network(self,model=0):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])
    b1 = self.bias_variable([20])
    W2 = self.weight_variable([20,self.action_dim])
    b2 = self.bias_variable([self.action_dim])
    if model != 0:
      print("model change!")
      [W1_old, b1_old, W2_old, b2_old] = self.build_net(model)
      W1 = W1_old
      b1 = b1_old
      W2 = W2_old
      b2 = b2_old
    # input layer
    self.state_input = tf.compat.v1.placeholder("float",[None,self.state_dim])
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2

  def create_training_method(self):
    self.action_input = tf.compat.v1.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.compat.v1.placeholder("float",[None])
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
    self.optimizer = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft()

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network()

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  # def egreedy_action(self,state,pg=0,no_action=10000):
  #   action = no_action
  #   Q_value = self.Q_value.eval(feed_dict = {
  #     self.state_input:[state]
  #     })[0]
  #   if random.random() <= self.epsilon:
  #       while action == no_action:
  #           # action = random.randint(0,self.action_dim - 1)
  #           action = pg % self.action_dim
  #       return action
  #       # return random.randint(0,self.action_dim - 1)
  #   else:
  #       if np.argmax(Q_value) == no_action:
  #           while action == no_action:
  #               action = random.randint(0,self.action_dim - 1)
  #           return action
  #       return np.argmax(Q_value)
  def epsilonc(self,e):
      if self.epsilon > FINAL_EPSILON:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/e
  def egreedy_action(self,state,next=0):
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    if random.random() <= self.epsilon:
      return random.randint(0,self.action_dim - 1)
    else:
      x = Q_value
      while next:
        x[np.argmax(x)] = np.min(x)
        next = next - 1
      return np.argmax(x)

    

  def action(self,state):
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    initial = tf.random.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

  def save_net(self, save_path):
    # saver = tf.compat.v1.train.Saver()
    self.saver.save(self.session, save_path, write_meta_graph=False)
    
    variable_names = [v.name for v in tf.trainable_variables()]
    values = self.session.run(variable_names)
    for k,v in zip(variable_names, values):
      print("Variable: ", k)
      print("Shape: ", v.shape)
      print(v)
    
    print("Save to path: ", save_path)

  def close(self):
    tf.compat.v1.reset_default_graph()
    self.session.close()

  def build_net(self, path, add=0):
    self.saver.restore(self.session, path)
    variable_names = [v.name for v in tf.trainable_variables()]
    values = self.session.run(variable_names)
    v_old = []
    for k,v in zip(variable_names, values):
      print("Variable: ", k)
      print("Shape: ", v.shape)
      print(v)
      v_old.append(v)
    return  v_old
    
    # print(self.sess.run(W1))  
                
