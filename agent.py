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

        self.memory_replay = MemoryReplay(10000)

        self.policy_net = DQN(len(Env.action_space)).to(device)
        self.target_net = DQN(len(Env.action_space)).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        self.gamma = 0.99
        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1


    def train(self):
        if len(self.memory_replay) < self.batch_size:
            return

        obs, actions, rewards, next_obs, terminals = self.memory_replay.sample(self.batch_size)

        obs = torch.FloatTensor(obs).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_obs = torch.FloatTensor(next_obs).to(device)
        terminals = torch.FloatTensor(terminals).to(device)

        # Current Q values
        q_values = self.policy_net(obs) #NOTE: doing policy_net(obs) actually calls policy_net's foward method since torch overwrites the __call__ method
        #q_values is a matrix [batch_size, num_actions], with a probability of choosing each action for each state in the matrix
        # -> then, .gather selects the max on each row
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        #Target Q values
        with torch.no_grad():
            # Note: The value of taking action a in state s equals the reward (from the env) plus the discounted best future value (which is predicted by the target network)
            # ie. Q(obs, action) = reward + gamma * target_net(next_obs) (doing target_net(next_obs) gets the action with the best reward out of all actions)
            next_q_values = self.target_net(next_obs)
            max_next_q = next_q_values.max(1)[0]

        target = rewards + self.gamma * max_next_q * (1 - terminals)

        loss = F.smooth_l1_loss(q_value, target) #here q_value is the value of the actual action taken, so we want to compute the loss of the actual action taken vs. the optimal action to have taken in that scenario

        self.optimizer.zero_grad() #resets gradient values, since in pytorch calling loss.backward twice stacks (ie. gradient = loss + loss)
        loss.backward() # performs back prop using autograd
        self.optimizer.step() # updates weights



        # play with random moves until memory buffer is filled to a certain capacity
        #   -> make predictions on reward with current state of network

        # then, begin randomly playing vs sampling from mem

        # need to implement epsilon-greedy here as well

        # Next action:
        # (feed the observation to your agent here)

        # action = env.action_space.sample()

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
            for episode in range(1000):
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

                        # In train_loop, replace self.train() with:
                        self.step_count += 1
                        if self.step_count % 4 == 0:  # only train every 4 steps
                            self.train()

                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                    if episode % 100 == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                    print("Episode:", episode, "Reward:", total_reward)
                except:
                    print("exception occured")

            self.target_net.save()


agent = Agent()
agent.train_loop()