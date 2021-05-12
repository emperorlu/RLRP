
from park.param import config
import numpy as np

servers = [0] * config.num_servers
reward = 0
for i in range(config.num_stream_jobs):
    reward -= np.std(servers)
    print("servers:",servers)