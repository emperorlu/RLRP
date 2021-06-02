#!/usr/bin/env python
# coding=utf-8
import park
from dqn import DQN
# from ddpg import DDPG
# from dqn2 import DeepQNetwork
from qlearning import QLearningTable
import pandas as pd 
import numpy as np
import tensorflow as tf
from park.param import config
import time
import matplotlib.pyplot as plt
# from ddpg import Actor, Critic
from memory import *
import warnings
warnings.filterwarnings("ignore")

EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 1 # The number of experiment test every 100 episode
Rnum = 3
final_map = []
osd = []

def DDPGLearn():
    env = park.make('data_migration')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high  # scale action, [-action_range, action_range]

    agent = DDPG(action_dim, state_dim, action_range)
    equ = 200

    for episode in range(EPISODE):
        snum = config.num_stream_jobs / (config.num_servers-1)
        snum = snum * config.num_rep
        serverss = [int(snum)] * config.num_servers
        serverss[config.num_servers-1] = 0
        state = env.reset(serverss)
        done = False
        i = 0
        MEMORY_CAPACITY = 10000 
        while not done:
            action = agent.get_action(state)
            state_, reward, done = env.step(action,i)
            i += 1
            # RL.learn(str(state), action, reward, str(state_))
            agent.store_transition(state,action,reward,state_)
            if agent.pointer > MEMORY_CAPACITY:
                agent.learn()
            state = state_
            if done:
                if np.std(state) < equ: 
                    equ = np.std(state)
                    print("Best Now!")
                    print("episode:",episode," state: ", state, "\nstd:",np.std(state), " epsilon:", agent.epsilon)
                if episode%100 == 0: 
                    print("episode:",episode," state: ", state, "\nstd:",np.std(state), " epsilon:", agent.epsilon)

def Mreward(map1,map2):
    num = 0
    for i in range(len(map1)):
        if map1[i] != map2[i]:
            num += 1
    return num

def hua(st,osd=0,osd_new=0,osd_zhu=0):
    plt.figure(10)
    x = range(len(st))
    plt.xlabel("episode")
    plt.ylabel("std")
    plt.title('Train')
    plt.plot(x, st)
    plt.savefig("pig/test_st.png")
 
    # plt.subplot(212)
    if osd_new == 0:
        plt.figure(11)
        x=range(len(osd))
        y=osd
        # xticks1=list(ppv3.index) 
        plt.bar(x,y)
        plt.xticks(x)
        plt.xlabel('OSD')
        plt.ylabel('PG number')
        plt.title('Placement')
        for a,b in zip(x,y):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        # plt.ylim(0,100)

        plt.savefig("pig/test_osd.png")
    elif osd_zhu == 0:
        plt.figure(11)
        # osd.append(0)
        x1=range(len(osd))
        x2=range(len(osd_new))
        y1= [osd[i] - osd_new[i] for i in range(len(osd_new))]
        y2=osd_new
        # xticks1=list(ppv3.index) 
        plt.bar(x2,y2)
        plt.bar(x1, y1, bottom=y2, label='move number')
        plt.xticks(x2)
        plt.xlabel('OSD')
        plt.ylabel('PG number')
        plt.title('Placement')
        # for a,b in zip(x1, y1):
        #     plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        for a,b in zip(x2, y2):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        # plt.ylim(0,100)

        plt.savefig("pig/test_osdnew.png")
    else:
        plt.figure(11)
        # osd.append(0)
        x1=range(len(osd))
        x2=range(len(osd_zhu))
        # osd[0] += 5
        # osd[4] -= 5
        y2= [osd[i] - osd_zhu[i] for i in range(len(osd_zhu))]
        y1= osd_zhu
        y3= osd
        # xticks1=list(ppv3.index) 
        plt.bar(x1,y3,color='darkorange')
        plt.bar(x1,y1,color='steelblue')
        
        # plt.bar(x2, y2, bottom=y1, label='move number')
        plt.xticks(x2)
        plt.xlabel('OSD')
        plt.ylabel('PG number')
        plt.title('Placement')
        for a,b in zip(x1, y3):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        for a,b in zip(x1, y1):
            plt.text(a, b+0.05, '%.0f' % b, ha='center', va= 'bottom',fontsize=7)
        # plt.ylim(0,100)

        plt.savefig("pig/test_osdzhu.png")

def Zhu():

    serverss = [0] * config.num_servers
    zhu = [0] * config.num_servers
    a=np.load('mapping.npy')
    osd=np.sum(a,axis=0)
    # data = [51,54,58,55,61,56,51,52,54,53,55]
    for pg_num in range(len(serverss)):
        serverss[pg_num] = int(osd[pg_num])
        # serverss[pg_num] = int(data[pg_num])
    rows,cols=a.shape
    # cols += 1
    print(rows," X ",cols)
    for i in range(rows):
        # num = random.randint(0,2)
        k = random.randint(0,cols-1)
        while(zhu[k]>21):
            k += 1
            k = k%cols
        zhu[k] += 1
        # for j in range(cols):
            
            # if a[i][j] == 1: 
            #     if num == 0: 
            #         k = j
            #         while(zhu[k]>22):
            #             k += 1
            #             k = k%cols
            #         zhu[k] += 1
            #         break
            #     else:
            #         num -= 1
    
    print("serverss: ", serverss)
    print("zhu: ", zhu)
    hua([],serverss,[],zhu)

def DQNLearnSigle3():
    env = park.make('replica_placement')
    agent = DQN(env)#,0.1)
    e = EPISODE / 10
    equ = 100
    st = []
    num = 0
    Rnum = 3
    t0 = time.time()
    i = 0
    for episode in range(EPISODE):
        i += 1
        state = env.reset()
        done = False
        while not done:
            j = 0
            Raction = []
            while j != Rnum:
                action = agent.egreedy_action(state)
                # print("action: ", action)
                if action not in Raction:
                    Raction.append(action)
                    next_state,reward,done = env.step(action)
                    agent.perceive(state,action,reward,next_state,done)
                    state = next_state
                    j += 1
            # action = agent.egreedy_action(state) 
            # next_state,reward,done = env.step(action)
            # state = next_state
            fstate = env.observe()
            # agent.perceive(state,action,reward,next_state,done)
            if (done):
                st.append(np.std(fstate))
                if np.std(fstate) < equ: 
                    equ = np.std(fstate)
                    print("Best Now!")
                if np.std(fstate) < 1: num += 1
                else: num = 0
                print("episode:",episode, "\nstd:",np.std(fstate), " epsilon:", agent.epsilon,"\nstate: ", state, "\nservers:", fstate)
        if num == 3: break
        agent.epsilonc(e)
    t1 = time.time()
    print("total episode:",i,"; cost time: ", t1-t0)
    hua(st,osd)
    agent.save_net("./dqn_model/place3.ckpt")
    agent.close()


def DQNLearnSigle():
    env = park.make('replica_placement')
    agent = DQN(env)#,0.1)
    e = EPISODE / 10
    equ = 100
    st = []
    num = 0
    t0 = time.time()
    i = 0
    for episode in range(EPISODE):
        i += 1
        state = env.reset()
        done = False
        j = 0
        num_ain = 0
        old_action = []
        while not done:
            action = agent.egreedy_action(state) 
            if j%config.num_rep == 0 : 
                old_action = []
                old_action.append(action) #副本
                next_state,reward,done,ain = env.step(action)
            else:
                next_state,reward,done,ain = env.step(action,0,old_action)
                old_action.append(action)
            j += 1
            if ain: num_ain += 1
            state = next_state
            fstate = env.observe()
            agent.perceive(state,action,reward,next_state,done)

            if (done):
                st.append(np.std(fstate))
                if np.std(fstate) < equ: 
                    equ = np.std(fstate)
                    print("Best Now!")
                if np.std(fstate) < 1 and num_ain == 0 : num += 1
                else: num = 0
                print("episode:",episode, " epsilon:", agent.epsilon, "\nstd:",np.std(fstate), " ain:", num_ain,"\nstate: ", state, "\nservers:", fstate)
        if num == 3: break
        agent.epsilonc(e)
    t1 = time.time()
    print("total episode:",i,"; cost time: ", t1-t0)
    hua(st,osd)
    agent.save_net("./dqn_model/place3.ckpt")
    agent.close()

def DQNTestSigle():
    env = park.make('replica_placement')
    agent = DQN(env,0)
    agent.build_net("./dqn_model/place3.ckpt")
    st = []
    ac = []
    for episode in range(TEST):
        state = env.reset()
        done = False
        t0 = time.time()
        num = int(config.num_stream_jobs * config.num_rep  / env.stepn)
        print("num: ",num)
        # while num:
        for i in range(num):
            state = env.reset(1)
            done = False
            while not done:
                action = agent.egreedy_action(state) 
                ac.append(action)
                next_state,reward,done = env.step(action)
                state = next_state
                # j = 0
                # Raction = []
                # while j != Rnum:
                #     action = agent.egreedy_action(state)
                #     if action not in Raction:
                #         Raction.append(action)
                #         next_state,reward,done = env.step(action)
                #         state = next_state
                #         j += 1
        fstate = env.observe()
        t1 = time.time()
        print("total episode:",i,"; cost time: ", t1-t0)
        print("episode:",episode, "\nstd:",np.std(state), " epsilon:", agent.epsilon,"\nstate: ", state, "\nservers:", fstate)
        # print("episode:",episode, "\nstd:",np.std(state), " epsilon:", agent.epsilon,"\nstate: ", state, "\nservers:", fstate)
        # agent.epsilonc(e)
    for pg_num in range(len(ac)):
        print(pg_num,"————>",ac[pg_num*3], ",", ac[pg_num*3+1], ",", ac[pg_num*3+2]) 
    hua(st,fstate)
    agent.close()


def DQNLearn():
  global osd
  global final_map
  global Rnum
  env = park.make('replica_placement')
  t0 = time.time()
  agent = DQN(env)
  equ = 100
  i = 0
  e = EPISODE * 2 / 3
  st = []
  for episode in range(EPISODE):
    i += 1 
    state = env.reset()
    done = False
    map = []
    while not done:
      i = 0
      Raction = []
      while i != Rnum:
          action = agent.egreedy_action(state)
          if action not in Raction:
              Raction.append(action)
              next_state,reward,done = env.step(action)
            #   print("i: ",i,"state:",state)
            #   print("reward: ",reward,"action: ",action)
              agent.perceive(state,action,reward,next_state,done)
              state = next_state
              i += 1
    #   action = agent.egreedy_action(state) # e-greedy action for train
    #   next_state,reward,done = env.step(action)
      map.append(Raction)
      if (done):
        st.append(np.std(state))
        if np.std(state) < equ: 
            equ = np.std(state)
            final_map = map
            osd = state
            print("Best Now!")
        print("episode:",episode," state: ", state, "\nstd:",np.std(state), " epsilon:", agent.epsilon)
    agent.epsilonc(e)

  mapping = np.zeros((config.num_stream_jobs, config.num_servers))
  t1 = time.time()
  print("total episode:",i,"; cost time: ", t1-t0)
  print("osd state:",osd)
  hua(st,osd)
  f = open("map1.txt", 'w+')
  for pg_num in range(len(final_map)):
    print(pg_num,"————>",final_map[pg_num], file=f)
    mapping[pg_num][final_map[pg_num]] = 1
  np.save('map.npy',np.array(final_map))
  np.save('mapping.npy',mapping)
    #   a=np.load('map.npy')
    #   a=a.tolist()
    #   for pg_num in range(len(a)):
    #     print(pg_num,"————>",a[pg_num])
    #   agent.save_net("./dqn_model/place.ckpt")
  agent.close()


def DQNTest():
    global osd
    global final_map
    print("----Test----")
    env = park.make('replica_placement')
    agent = DQN(env)
    # map=np.load('map.npy')
    # map=map.tolist()
    map = [0] * 1000
    for pg_num in range(len(map)):
        map[pg_num] = pg_num % 10
        print(pg_num,"————>",map[pg_num])
    # agent.build_net("./dqn_model/place.ckpt")
    # map1 = []
    e = EPISODE / 10
    r = 1000000000
    for episode in range(EPISODE):
        state = env.reset()
        map2 = []
        done = False
        i = 0
        num = 0

        while not done:
            old_action = map[i]
            action = agent.egreedy_action(state,old_action) # e-greedy action for train
            i += 1
            # map1.append(action)
            equ = 0
            map2.append(action)
            if old_action == action: equ=1
            next_state,reward,done = env.step(action,equ)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                for pg_num in range(len(map)):
                    if map[pg_num] != map2[pg_num]: num += 1
                if (num+20) * np.std(state) < r: 
                    r = num * np.std(state)
                    final_map = map2
                    osd = state
                    print("best now!")
            # print("equ:",equ)
                print("episode:",episode," state: ", state, "\nstd:",np.std(state), " num:",num, " epsilon:", agent.epsilon)
                # print("episode:",episode)
                # print("state:",state)
                # print("reward:",reward)
            # agent.perceive(state,action,reward,next_state,done)
        agent.epsilonc(e)
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


def QlearningLearn_data():
    env = park.make('data_migration')
    RL = QLearningTable(env.action_space.n)
    # agent = DQN(env)
    equ = 200
    e = EPISODE * 3 / 4
    # snum = config.num_stream_jobs / (config.num_servers-1)
    # snum = snum * config.num_rep
    # serverss = [int(snum)] * config.num_servers
    serverss = [0] * config.num_servers
    a=np.load('mapping.npy')
    # a=a.tolist()
    osd=np.sum(a,axis=0)
    for pg_num in range(len(osd)):
        serverss[pg_num] = int(osd[pg_num])
    print("serverss: ", serverss)
    st = []
    osd_new = []
    # serverss[config.num_servers-1] = 0
    for episode in range(EPISODE):
        
        state = env.reset(serverss)
        done = False
        i = 0
        while not done:
            action = RL.choose_action(str(state))
            # action = agent.egreedy_action(state)
            state_, reward, done = env.step(action,i)
            i += 1
            RL.learn(str(state), action, reward, str(state_))
            # agent.perceive(state,action,reward,state_,done)
            state = state_
            if done:
                st.append(np.std(state))
                if np.std(state) < equ: 
                    equ = np.std(state)
                    osd_new = state[:]
                    print("Best Now!")
                    # print("episode:",episode," state: ", state, "\nstd:",np.std(state), " epsilon:", agent.epsilon)
                # if episode%100 == 0: 
                print("episode:",episode," state: ", state, "\nstd:",np.std(state))#, " epsilon:", agent.epsilon)
                # print("episode:",episode)
                # print("state:",state)
                # print("action:",action, "; reward:",reward) 
        # agent.epsilonc(e)
    print("osd: ",serverss,";\nosd_new:",osd_new,";\nst:",st)
    hua(st,serverss,osd_new)
    # agent.save_net("./dqn_model/move_less.ckpt")
    # agent.close()
    

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

def QlearningTest2():
    global osd
    global final_map
    print("----Test----")
    env = park.make('replica_placement')
    RL = QLearningTable(env.action_space.n)
    # map=np.load('map.npy')
    # map=map.tolist()
    map = [0] * config.num_stream_jobs
    for pg_num in range(len(map)):
        map[pg_num] = pg_num % 10
        print(pg_num,"————>",map[pg_num])
    # agent.build_net("./dqn_model/place.ckpt")
    # map1 = []

    r = 1000000000
    for episode in range(EPISODE):
        state = env.reset()
        map2 = []
        done = False
        i = 0
        num = 0
        while not done:
            old_action = map[i]
            action = RL.choose_action(str(state),old_action) # e-greedy action for train
            i += 1
            # map1.append(action)
            equ = 0
            map2.append(action)
            if old_action == action: equ=1
            next_state,reward,done = env.step(action,equ)
            RL.learn(str(state), action, reward,str(next_state))
            state = next_state
            if done:
                for pg_num in range(len(map)):
                    if map[pg_num] != map2[pg_num]: num += 1
                if num * np.std(state) < r: 
                    r = num * np.std(state)
                    final_map = map2
                    osd = state
                    print("best now!")
            # print("equ:",equ)
                print("episode:",episode," state: ", state, "\nstd:",np.std(state), " num:",num)
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

if __name__ == '__main__':
    
    # DQNTest()
    print("begin train for placement\n")
    # DQNLearn()
    # print("begin test\n")
    # QlearningLearn_data()
    # Zhu()
    DQNLearnSigle()
    DQNTestSigle()
    # QlearningLearn()
    # QlearningTest()




