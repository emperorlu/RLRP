import numpy as np

from park import core, spaces, logger
from park.param import config
from park.utils import seeding

class ReplicaplacementEnv(core.Env):

    def __init__(self):

        self.setup_space()
        self.seed(config.seed)
        self.num_stream_jobs = config.num_stream_jobs
        self.servers = self.initialize_servers()
        self.reset()

    def initialize_servers(self):
        servers = [0] * config.num_servers
        return servers

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
        self.action_space = spaces.Discrete(config.num_servers)

    def step(self, action):

        # 0 <= action < num_servers
        std1 = np.std(self.servers)
        assert self.action_space.contains(action)
        self.servers[action] = self.servers[action] + 1
        std2 = np.std(self.servers)
        reward = 0
        reward = std1 - std2
        # reward -= np.std(self.servers)
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        return self.observe(), reward, done

    def r_step(self, actions):
        std1 = np.std(self.servers)
        for action in actions:
            print("action: ",action)
            for a in self.action_space:
                print(a)
            assert self.action_space.contains(action)
            self.servers[action] = self.servers[action] + 1
        std2 = np.std(self.servers)
        reward = 0
        reward = std1 - std2
        # reward -= np.std(self.servers)
        # reward = min(self.servers) - max(self.servers)

        self.num_stream_jobs_left = self.num_stream_jobs_left - 1
        done = (self.num_stream_jobs_left == 0)
        return self.observe(), reward, done