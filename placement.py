#!/usr/bin/env python
# coding=utf-8
import park
from dqn import DQN
from qlearning import QLearningTable

EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

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
    #   print("state:",state)
    #   print("act:",action)
    #   print("reward:",reward)
      # Define reward for agent
      reward_agent = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done)
      state = next_state
    #   if done:
    #     break
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
        #   env.reset()
          action = agent.action(state) # direct action for test
          state,reward,done = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)
      if ave_reward >= 0:
        break

def QlearningTest():
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        while not done:
            action = RL.choose_action(str(state))
            state_, reward, done = env.step(action)
            print("state:",state)
            print("act:",action)
            print("reward:",reward)
            # RL learn from this transition
            RL.learn(str(state), action, reward, str(state_))
            state = state_

    # end of game
    print('game over')
    print(RL.q_table)
    env.destroy()


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




if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()