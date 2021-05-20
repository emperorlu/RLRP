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

EPISODE = 10 # Episode limitation
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
    action_bound = config.load_balance_obs_high

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
  global osd
  global final_map
  env = park.make('replica_placement')
#   osd_num = config.num_servers_now
  agent = DQN(env)
  equ = 100
  for episode in range(EPISODE):
    state = env.reset()
    done = False
    map = []
    while not done:
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done = env.step(action)
      map.append(action)
      
      # Define reward for agent
    #   reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if (done):
        if np.std(state) < equ: 
            equ = np.std(state)
            final_map = map
            osd = state
            # print(final_map)
            print("state:",state," sum:",sum(state),"equ:",equ)
            # print("equ:",equ)
        print("episode:",episode," std:",np.std(state))
        # print("episode:",episode)
        # print("state:",state)
        # print("reward:",reward)
#   a=np.array(a)
  print("osd state:",osd)
  for pg_num in range(len(final_map)):
    print(pg_num,"————>",final_map[pg_num])
  np.save('map.npy',np.array(final_map))
  a=np.load('map.npy')
  a=a.tolist()
  for pg_num in range(len(a)):
    print(pg_num,"————>",a[pg_num])
#   agent.save_net("./dqn_model/place.ckpt")
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
    global osd
    global final_map
    print("----Test----")
    env = park.make('replica_placement')
    agent = DQN(env)
    map=np.load('map.npy')
    map=map.tolist()
    # agent.build_net("./dqn_model/place.ckpt")
    # map1 = []
    map2 = []
    r = -100
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        i = 0
        while not done:
            action = agent.egreedy_action(state) # e-greedy action for train
            old_action = map[i]
            i += 1
            # map1.append(action)
            equ = 0
            map2.append(action)
            if old_action == action: equ=1
            next_state,reward,done = env.step(action,equ)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                if reward > r: 
                    r = reward
                    final_map = map2
                    osd = state
                    print("best now!")
            # print("equ:",equ)
                print("episode:",episode," std:",reward)
                # print("episode:",episode)
                # print("state:",state)
                # print("reward:",reward)
            # agent.perceive(state,action,reward,next_state,done)
    print("osd state:",osd)
    num = 0
    for pg_num in range(len(map)):
        print(pg_num,"————>",map[pg_num],"; ",pg_num,"————>",final_map[pg_num])
        if map[pg_num] != final_map[pg_num]: num += 1
    print("different pgs: ", num)
    # print 
    # for pg_num in range(len(final_map)):
    #     print(pg_num,"————>",final_map[pg_num])        
    # for episode in range(EPISODE):
    #     # map2 = []
    #     no_action = 1
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.egreedy_action(state) # e-greedy action for train
    #         # map1.append(action)
    #         next_state,reward,done = env.step(action)
    #         if done:
    #             print("state:",state)
    #             print("reward:",reward)
    #         # agent.perceive(state,action,reward,next_state,done)
    #         state = next_state
    #     while not done:
    #         action = agent.egreedy_action(state,no_action) # e-greedy action for train
    #         # map2.append(action)
    #         #   print("action:",action)
    #         next_state,reward,done = env.step(action)#, Mreward(map1,map2))
    #         if done:
    #             print("episode:",episode)
                
    #             print("state:",state)
    #             print("reward:",reward)
    #         # agent.perceive(state,action,reward,next_state,done)
    #         state = next_state
    #     next_state,reward,done = env.Mstep(action,Mreward)
    #     print("Mreward(map1,map2):",Mreward(map1,map2))
    #     agent.perceive(state,action,reward,next_state,done)
    agent.close()

def QlearningLearn():
    global osd
    global final_map
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    # equ = 100
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        # map = []
        while not done:
            # Raction = []
            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)
            # i = 0
            # while i != Rnum:
            #     action = RL.choose_action(str(state))
            #     if action not in Raction:
            #         Raction.append(action)
            #         i += 1
            
            # map.append(Raction)
            # state_, reward, done = env.r_step(Raction)
            
                # print("state:",state," sum:",sum(state))
                # print("act:",action)
                # print("reward:",reward)
            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))
            if done:
                print("episode:",episode)
                print("state:",state)
                print("reward:",reward)
            state = state_
        #     if (done):
        #         if np.std(state) < equ: 
        #             equ = np.std(state)
        #             final_map = map
        #             osd = state
        #             # print(final_map)
        #             print("state:",state," sum:",sum(state),"equ:",equ)
        #             # print("equ:",equ)
        #         print("episode:",episode," std:",np.std(state))
        # if equ == 0:
        #     # print("Perfect mapping!")
        #     break
    # end of game
    # print('train over')
    # print("final state:",osd,"equ:",equ)
    # # RL.model_saver('q-learning.pkl')
    # print("mapping:")
    # for pg_num in range(len(final_map)):
    #     print(pg_num,"————>",final_map[pg_num])
    # f = open("map1.txt", 'w+')
    # for pg_num in range(len(final_map)):
    #     print(pg_num,"————>",final_map[pg_num], file=f)
    
    print(RL.q_table)
    save = pd.DataFrame(RL.q_table) 
    save.to_csv('ql.csv')  #index=False,header=False表示不保存行索引和列标题
    env.close()
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
    DQNTest()
    # QlearningLearn()
    # QlearningTest()
    # DDPGLearn()
    

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


