import numpy as np
import pandas as pd
# import pickle

class QLearningTable:
     def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = range(actions)  # a list动作集
        self.lr = learning_rate #学习率
        self.gamma = reward_decay #回报的衰减系数
        self.epsilon = e_greedy #贪婪系数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #定义q表

     def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

     def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

     def check_state_exist(self, state):
         if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

     def model_saver(self,path):
         self.q_table.to_csv(path)
        #  with open(path, 'wb') as f:
        #     pickle.dump(dict(self.q_table), f)
         print(self.q_table)
         print('model saved')

     def model_loader(self, path):
         self.q_table = pd.read_csv(path,index_col=0,dtype=np.float64)
         print(self.q_table)
         print('model loaded')
        #  with open(path, 'rb') as f:
        #     self.q_table = pickle.load(f)
            

