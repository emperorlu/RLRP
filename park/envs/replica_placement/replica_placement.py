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
        self.servers = self.initialize_servers()
        self.reset()

    def initialize_servers(self):
        servers = [0] * config.num_servers
        return servers
    
    def set_servers(self, ser):
        self.servers = ser

    def observe(self):
        # obs_arr = []
        # # load on each server
        # for server in self.servers:
        #     obs_arr.append(server)
        # obs_arr = np.array(obs_arr)
        self.observation_space.contains(self.servers)

        return self.servers

    def reset(self):
        # for server in self.servers:
        #     server.reset()
        self.servers = self.initialize_servers()
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

    def step(self, action):

        # 0 <= action < num_servers
        # std1 = np.std(self.servers)
        assert self.action_space.contains(action)
        self.servers[action] = self.servers[action] + 1
        # std2 = np.std(self.servers)
        reward = 0
        # reward = std1 - std2
        if (np.std(self.servers) == 0): reward = 10000
        # bei = 10
        # if equ == 1: bei =1
        reward -= np.std(self.servers) ** 0.5
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        return self.observe(), reward, done

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
        self.servers = self.initialize_servers()
        self.reset()

    def initialize_servers(self, state_current=0):
        if state_current == 0:
            return [0] * config.num_servers
        servers = state_current[:]
        return servers
    
    def set_servers(self, ser):
        self.servers = ser

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
        self.action_space = spaces.Discrete(config.num_rep+1)

    def step(self, action, i=0):
        std1 = np.std(self.servers)
        assert self.action_space.contains(action)
        if action != 3:
            action = (i+action) % (config.num_servers-1)
            if self.servers[action] > 0:
                self.servers[action] = self.servers[action] - 1
                self.servers[config.num_servers-1] = self.servers[config.num_servers-1] + 1
        
        std2 = np.std(self.servers)
        # reward = 1000
        c =  std2 - std1
        reward = -np.std(self.servers) **0.5
        # else: reward -= np.std(self.servers) #* (num+1)
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        if np.std(self.servers) < 3 and c > 0: done = True
        if self.servers[-1] >= self.men: done = True
        # if self.servers[-1] == max(self.servers): done = True
        return self.observe(), reward, done