#!/usr/bin/env python
# coding=utf-8
import park

env = park.make('replica_placement')

#obs = env.reset()
#done = False

for i_episode in range(20):
    #print(obs)
    print("Episode finished after {} timesteps".format(i_episode+1))
    obs = env.reset()
    #print(obs)
    done = False   
    while not done:
        print("obs:\n",obs)
        
        # act = agent.get_action(obs)
        act = env.action_space.sample()
        print("act:\n",act)
        obs, reward, done = env.step(act)
        print("act:\n",act)
        print("reward:\n",reward)
   # print("Episode finished after {} timesteps".format(i_episode+1))
    #print(obs)
    #print(act)
    #print(reward)
