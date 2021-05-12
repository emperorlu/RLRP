#!/usr/bin/env python
# coding=utf-8
import park
from dqn import DQN
from qlearning import QLearningTable
import pandas as pd 
import numpy as np

EPISODE = 20 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
Rnum = 2

def DQNTest():

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
        print("act:",action)
        print("reward:",reward)
      # Define reward for agent
      reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
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

def QlearningTest():
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    equ = 100
    final_map = []
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        steps = 0
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
            
            if (episode%1 == 0 and done):
                if np.std(state) < equ: 
                    equ = np.std(state)
                    final_map = map
                    print(final_map)
                    print("state:",state," sum:",sum(state))
                    print("act:",action)
                    print("reward:",reward)
                print("episode:",episode," step:",steps)
                # print("state:",state," sum:",sum(state))
                # print("act:",action)
                # print("reward:",reward)
            steps += 1
            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))
            state = state_
        
        if equ == 0:
            print("Perfect mapping!")
            break

    # end of game
    print('game over')
    print("final_map:",final_map)
    print(RL.q_table)
    save = pd.DataFrame(RL.q_table) 
    save.to_csv('ql.csv')  #index=False,header=False表示不保存行索引和列标题
    # env.destroy()


if __name__ == '__main__':
    # DQNTest()
    QlearningTest()
    




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


