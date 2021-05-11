#!/usr/bin/env python
# coding=utf-8
import park
from dqn import DQN

EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():

  env = park.make('replica_placement')
  agent = DQN(env)

  for episode in xrange(EPISODE):
    # initialize task
    state = env.reset()
    # Train
    for step in xrange(STEP):
      action = agent.egreedy_action(state) # e-greedy action for train
      next_state,reward,done,_ = env.step(action)
      # Define reward for agent
      reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
      if done:
        break
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in xrange(TEST):
        state = env.reset()
        for j in xrange(STEP):
          env.render()
          action = agent.action(state) # direct action for test
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      if ave_reward >= 200:
        break

if __name__ == '__main__':
    main()



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

