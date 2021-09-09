import numpy as np

from park import core, spaces, logger
from park.param import config
from park.utils import seeding

class ReplicaplacementEnv(core.Env):

    def __init__(self):

        self.setup_space()
        self.seed(config.seed)
        self.num_stream_jobs = config.num_stream_jobs
        # self.cur_servers = config.num_servers_now
        self.servers_state = self.initialize_servers(0)
        self.servers = self.initialize_servers(0)
        # self.weight = [21,35,5,44,7,54,25,12,2,22]
        self.weight = [1] * config.num_servers
        # self.Hash = np.zeros(config.num_stream_jobs)
        # for hi in range(1000000):
        #     hj = hi % config.num_stream_jobs
        #     self.Hash[hj] = self.Hash[hj] + 1
        # print("hash: ", self.Hash)
        # self.weight = [2, 3, 5, 4, 7, 2, 2, 2, 2, 2] 
        self.reset()

    def initialize_servers(self,old):
        if old == 0: servers = [0] * config.num_servers
        else: servers = [old[i] for i in range(len(old))]
        return servers
    
    def set_servers(self, ser):
        self.servers = ser

    def observe_state(self):
        #fstate = [self.servers[i] - min(self.servers) for i in range(len(self.servers))]
        #self.servers_state = [fstate[i] / self.weight[i] for i in range(len(fstate))]
        fstate = [self.servers[i] / self.weight[i] for i in range(len(self.servers))]
        self.servers_state = [round(fstate[i] - min(fstate)) for i in range(len(fstate))]
        return self.servers_state
    def observe(self):
        # obs_arr = []
        # # load on each server
        # for server in self.servers:
        #     obs_arr.append(server)
        # obs_arr = np.array(obs_arr)
        self.observation_space.contains(self.servers)

        return self.servers

    def reset(self,test=0,old=0):
        # for server in self.servers:
        #     server.reset()
        if test==0: self.servers = self.initialize_servers(old)
        self.stepn = 10 * config.num_servers
        self.num_stream_jobs_left = self.num_stream_jobs * config.num_rep
        assert self.num_stream_jobs_left > 0
        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.observation_space = spaces.Discrete(config.num_servers)
        self.action_space = spaces.Discrete(config.num_servers)

    def step(self, action, test=0, hnum=0):

        # 0 <= action < num_servers
        # std1 = np.std(self.servers)
        # print("action: ", action)
        assert self.action_space.contains(action)
        
        state = self.servers_state
        minn = 0;  maxn = 0
        self.servers[action] = self.servers[action] + 1 #self.Hash[hnum]
        if min(self.servers_state) == self.servers_state[action]: minn = 1
        if max(self.servers_state) == self.servers_state[action]: maxn = 1
        # state[action] = state[action] + 1/self.weight[action]
        state = self.observe_state()
        reward = 0
        if (np.std(self.servers) == 0): 
            reward = 10000
            done = True
        reward -= np.std(state) #** 0.5
        if minn: reward = reward + 100
        if maxn: reward = reward - 100
            
        # print("reward: ", reward)
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        self.stepn = self.stepn - 1
        if test == 0:done = (self.stepn == 0)
        else: done = (self.num_stream_jobs_left == 0)
        # done = (self.stepn == 0)
        return state, reward, done
        # return self.observe(), reward, done

    def r_step(self, actions):
        std1 = np.std(self.servers)
        for action in actions:
            assert self.action_space.contains(action)
            self.servers[action] = self.servers[action] + 1
        std2 = np.std(self.servers)
        reward = 0
        reward = std1 - std2
        if (np.std(self.servers) == 0): reward = 10000
        # reward -= np.std(self.servers) * 100
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        return self.observe(), reward, done

class DatamigrationEnv(core.Env):

    def __init__(self):

        # servers = [300] * config.num_servers
        # servers[config.num_servers-1] = 0
        # print("state:",servers)
        self.setup_space()
        self.seed(config.seed)
        self.num_stream_jobs = config.num_stream_jobs
        self.men = int(config.num_stream_jobs * config.num_rep / config.num_servers)
        print("men=",self.men)
        self.servers_state = self.initialize_servers()
        self.servers = self.initialize_servers()
        self.reset()

    def initialize_servers(self, state_current=0):
        if state_current == 0:
            return [0] * config.num_servers
        servers = state_current[:]
        return servers
    
    def set_servers(self, ser):
        self.servers = ser

    def observe_state(self):
        m = min(self.servers[:-1])
        for i in range(len(self.servers)-1):
            self.servers_state[i] = self.servers[i] - m
    
        return self.servers_state

    def observe(self):
        # obs_arr = []
        # # load on each server
        # for server in self.servers:
        #     obs_arr.append(server)
        # obs_arr = np.array(obs_arr)
        self.observation_space.contains(self.servers)

        return self.servers

    def reset(self,state_current=0):
        # for server in self.servers:
        #     server.reset()
        if state_current != -1:
            self.servers = self.initialize_servers(state_current)
        self.num_stream_jobs_left = self.num_stream_jobs 
        assert self.num_stream_jobs_left > 0
        return self.observe()

    def seed(self, seed):
        self.np_random = seeding.np_random(seed)

    def setup_space(self):
        # Set up the observation and action space
        # The boundary of the space may change if the dynamics is changed
        # a warning message will show up every time e.g., the observation falls
        # out of the observation space
        self.observation_space = spaces.Discrete(config.num_servers)
        self.action_space = spaces.Discrete(config.num_rep)

    def step(self, action, i=0):
        # std1 = np.std(self.servers)
        assert self.action_space.contains(action)
        
        state = self.servers_state
        if action < 3:
            action = (i+action) % (config.num_servers-1)
            if self.servers[action] > 0:
                self.servers[action] = self.servers[action] - 1
                self.servers[config.num_servers-1] = self.servers[config.num_servers-1] + 1
        
        # state = self.observe_state()
        # std2 = np.std(self.servers)
        # reward = 1000
        reward = -np.std(self.servers) ** 0.5
        if max(state) == state[action]: reward = -reward
        # reward = std1 - std2
        # else: reward -= np.std(self.servers) #* (num+1)
        # reward = (min(self.servers) - max(self.servers)) ** 0.5

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        # if np.std(self.servers) < 1: done = True
        if self.servers[-1] >= np.mean(self.servers): done = True
        # if self.servers[-1] == max(self.servers): done = True
        # print("self.observe_state():",self.observe_state())
        return self.observe_state(), reward, done
        # return self.observe(), reward, done