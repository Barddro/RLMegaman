import torch
from torch import optim

from dqn import *

from env import Env
from memory_replay import MemoryReplay
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self):
        self.step_count = 0
        self.env = Env()
        self.env.reset()

        self.memory_replay = MemoryReplay(100000)
        self.warmup_steps = 1000

        self.policy_net = DQN(len(Env.action_space)).to(device)
        self.target_net = DQN(len(Env.action_space)).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1

    def train(self):
        if len(self.memory_replay) < self.warmup_steps:
            return

        obs, actions, rewards, next_obs, terminals = self.memory_replay.sample(self.batch_size)

        obs = torch.FloatTensor(obs).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        terminals = torch.FloatTensor(terminals).to(device)

        q_values = self.policy_net(obs)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_obs)
            max_next_q = next_q_values.max(1)[0]

        target = rewards + self.gamma * max_next_q * (1 - terminals)

        loss = F.smooth_l1_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

    def select_action(self, obs):
        if random.random() < self.epsilon:
            action = random.randrange(len(self.env.action_space))
            #print(f"picking random action: {action}")
            return action

        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)

        with torch.no_grad():
            q_values = self.policy_net(obs)

        return q_values.argmax(dim=1).item()

    def train_loop(self):
        for episode in range(2400):
            try:
                obs = self.env.reset()
                terminal = False
                total_reward = 0

                while not terminal:
                    action = self.select_action(obs)
                    obs, reward, next_obs, terminal = self.env.step(action)

                    self.memory_replay.append(obs, action, reward, next_obs, terminal)

                    obs = next_obs
                    total_reward += reward

                    self.step_count += 1
                    if self.step_count % 4 == 0:
                        self.train()

                    if self.step_count % 10000 == 0:  # ← step-based, not episode-based
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                print("Episode:", episode, "Reward:", total_reward)

            except Exception as e:  # ← actually see your errors
                import traceback
                traceback.print_exc()

        self.target_net.save()


agent = Agent()
agent.train_loop()