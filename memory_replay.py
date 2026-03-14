import random
import numpy as np

class MemoryReplay:
    def __init__(self, capacity, obs_shape=(4, 84, 84)):
        self.capacity = capacity
        self.size = 0
        self.appendspot = 0

        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.terminals = np.zeros(capacity, dtype=np.float32)

    def append(self, obs, action, reward, next_obs, terminal):
        i = self.appendspot
        self.obs[i]       = obs
        self.next_obs[i]  = next_obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.terminals[i] = terminal

        self.appendspot = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.terminals[indices],
        )

    def __len__(self):
        return self.size