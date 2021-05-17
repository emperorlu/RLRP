#!/usr/bin/env python
# coding=utf-8
import park
from dqn import DQN
# from dqn2 import DeepQNetwork
from qlearning import QLearningTable
import pandas as pd 
import numpy as np
import tensorflow as tf
from park.param import config
import sys
# sys.path.append('Reinforcement-Learning/Deep_Deterministic_Policy_Gradient/')
from ddpg import Actor, Critic
from memory import *

EPISODE = 1000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
Rnum = 3
final_map = []
osd = []

def DDPGLearn():
    env = park.make('replica_placement')
    # env = env.unwrapped

    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    # action_bound = env.action_space.high
    action_bound = np.array([config.load_balance_obs_high] * (config.num_servers + 1))

    var = 3.

    with tf.Session() as sess:
        memory = Memory(32, 10000)
        actor = Actor(sess, state_dim, action_bound, lr=0.01, tau=0.01)
        critic = Critic(sess, state_dim, actor.s, actor.s_, actor.a, actor.a_, gamma=0.9, lr=0.001, tau=0.01)
        t = critic.get_gradients()

        actor.generate_gradients(t)

        sess.run(tf.global_variables_initializer())

        for episode in range(1000):
            s = env.reset()
            # r_episode = 0
            done = False
            # for j in range(200):
            while not done:
                a = actor.choose_action(s)
                print("a1:",a)
                a = np.clip(np.random.normal(a, var), -action_bound, action_bound)  # 异策略探索
                print("a2:",a)
                s_, r, done = env.step(a)

                memory.store_transition(s, a, [r / 10], s_)

                if memory.isFull:
                    var *= 0.9995
                    b_s, b_a, b_r, b_s_ = memory.get_mini_batches()
                    critic.learn(b_s, b_a, b_r, b_s_)
                    actor.learn(b_s)

                # r_episode += r
                
                if (done):
                    print("episode:",episode)
                    print("state:",s)
                    print("reward:",r)
                s = s_
                # if(j == 200 - 1):
                #     print('episode {}\treward {:.2f}\tvar {:.2f}'.format(i, r_episode, var))
                #     break

def Mreward(map1,map2):
    num = 0
    for i in range(len(map1)):
        if map1[i] != map2[i]:
            num += 1
    return num

def DQNLearn():

  env = park.make('replica_placement')
  agent = DQN(env)

  for episode in range(EPISODE):
    # initialize task
    state = env.reset()
    done = False
    # Train
    # print("state:\n",state)
    # for step in range(STEP):
    while not done:
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done = env.step(action)
      if (episode%1 == 0 and done):
        print("episode:",episode)
        print("state:",state)
        print("reward:",reward)
      # Define reward for agent
    #   reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
  agent.save_net("./dqn_model/place.ckpt")
  agent.close()
    #   if done:
    #     break
    # Test every 100 episodes
    # if episode % 100 == 0:
    #   total_reward = 0
    #   for i in range(TEST):
    #     state = env.reset()
    #     for j in range(STEP):
    #     #   env.reset()
    #       action = agent.action(state) # direct action for test
    #       state,reward,done = env.step(action)
    #       total_reward += reward
    #       if done:
    #         break
    #   ave_reward = total_reward/TEST
    #   print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
    #   if ave_reward >= 0:
    #     break

def DQNTest():
    print("----Test----")
    env = park.make('replica_placement')
    agent = DQN(env)
    agent.build_net("./dqn_model/place.ckpt")
    map1 = []
    map2 = []
    # for episode in range(EPISODE):
    state = env.reset()
    done = False
    while not done:
        action = agent.egreedy_action(state) # e-greedy action for train
        map1.append(action)
        next_state,reward,done = env.step(action)
        if done:
            print("state:",state)
            print("reward:",reward)
        # agent.perceive(state,action,reward,next_state,done)
        state = next_state
    for episode in range(EPISODE):
        map2 = []
        no_action = 1
        state = env.reset()
        done = False
        while not done:
            action = agent.egreedy_action(state) # e-greedy action for train
            map1.append(action)
            next_state,reward,done = env.step(action)
            if done:
                print("state:",state)
                print("reward:",reward)
            # agent.perceive(state,action,reward,next_state,done)
            state = next_state
        while not done:
            action = agent.egreedy_action(state,no_action) # e-greedy action for train
            map2.append(action)
            #   print("action:",action)
            next_state,reward,done = env.step(action)#, Mreward(map1,map2))
            if done:
                print("episode:",episode)
                
                print("state:",state)
                print("reward:",reward)
            # agent.perceive(state,action,reward,next_state,done)
            state = next_state
        next_state,reward,done = env.Mstep(action,Mreward)
        print("Mreward(map1,map2):",Mreward(map1,map2))
        agent.perceive(state,action,reward,next_state,done)
    agent.close()

def QlearningLearn():
    global osd
    global final_map
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    equ = 100
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        map = []
        while not done:
            Raction = []
            i = 0
            while i != Rnum:
                action = RL.choose_action(str(state))
                if action not in Raction:
                    Raction.append(action)
                    i += 1
            # state_, reward, done = env.step(action)
            map.append(Raction)
            state_, reward, done = env.r_step(Raction)
            
                # print("state:",state," sum:",sum(state))
                # print("act:",action)
                # print("reward:",reward)
            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))
            state = state_
            if (episode%1 == 0 and done):
                if np.std(state) < equ: 
                    equ = np.std(state)
                    final_map = map
                    osd = state
                    # print(final_map)
                    print("state:",state," sum:",sum(state),"equ:",equ)
                    # print("equ:",equ)
                print("episode:",episode," std:",np.std(state))
        if equ == 0:
            # print("Perfect mapping!")
            break
    # end of game
    print('train over')
    print("final state:",osd,"equ:",equ)
    RL.model_saver('q-learning.pkl')
    print("mapping:")
    for pg_num in range(len(final_map)):
        print(pg_num,"————>",final_map[pg_num])
    f = open("map1.txt", 'w+')
    for pg_num in range(len(final_map)):
        print(pg_num,"————>",final_map[pg_num], file=f)
    
    # print(RL.q_table)
    # save = pd.DataFrame(RL.q_table) 
    # save.to_csv('ql.csv')  #index=False,header=False表示不保存行索引和列标题
    # env.destroy()
def QlearningTest():
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    RL.model_loader('q-learning.pkl')
    Raction = []
    final_map = []
    i = 0
    # state = env.set_servers(osd)
    state = env.reset()
    done = False
    while not done:
        while i != Rnum:
            action = RL.choose_action(str(state))
            if action not in Raction:
                Raction.append(action)
                i += 1
        final_map.append(Raction)
        state_, reward, done = env.r_step(Raction)
        print("state_: ", state_, "reward: ", reward)
        # RL.learn(str(state), action, reward, str(state_))
        state = state_
    f = open("map2.txt", 'w+')
    for pg_num in range(len(final_map)):
        print(pg_num,"————>",final_map[pg_num], file=f)
    
    print("state:",state_," sum:",sum(state_))

if __name__ == '__main__':
    # DQNLearn()
    # DQNTest()
    # QlearningLearn()
    # QlearningTest()
    DDPGLearn()
    

    # "./dqn_model/placement.ckpt"




# for i_episode in range(20):
#     #print(obs)
#     print("Episode finished after {} timesteps".format(i_episode+1))
#     obs = env.reset()
#     #print(obs)
#     done = False   
#     while not done:
#         print("obs:\n",obs)
        
#         # act = agent.get_action(obs)
#         act = env.action_space.sample()
#         obs, reward, done = env.step(act)
#         # print("act:\n",act)
#         print("reward:\n",reward)


