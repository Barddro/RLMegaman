#old version of env using cynes
import math
import random
from collections import deque

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
from cynes import *
from cynes.windowed import WindowedNES
#from collectstatesauto import loadState
import cv2
from pathlib import Path

#https://datacrystal.tcrf.net/w/index.php?title=Mega_Man_2/RAM_map

def processSingleFrame(frame):
    '''
    Input:
    - Single 256 X 240 RGB frame as a 256 X 240 X 3 Tensor
    Output:
        - Downsampled 84 x 84 Grayscale frame
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    return resized.astype(np.uint8)


def getSizeOfDir(path):
    count = sum(1 for _ in path.iterdir())
    return count

def getRandomState(path_string="./"):
    path = Path(f"{path_string}/states")
    stages = [item for item in path.iterdir()]
    path = random.choice(stages)

    num_of_substates = getSizeOfDir(path)

    return f"{str(path)}/{random.randrange(num_of_substates)}.state"


class Env:
    #Registers
    SCREENID = 0x0440
    HP = 0x06C0
    ENEMIES_START = 0x06C2
    ENEMIES_END = 0x06CF
    BOSS_HP = 0x06C1
    X_POS = 0x0460
    Y_POS = 0x04A0
    LIVES = 0x00A8

    #Action Space
    action_space = [
        0, #no-op
        NES_INPUT_LEFT,
        NES_INPUT_RIGHT,
        NES_INPUT_UP,
        NES_INPUT_DOWN,
        NES_INPUT_A,
        NES_INPUT_B,
        NES_INPUT_RIGHT | NES_INPUT_B
    ]

    #Reward Params
    SCREEN_REWARD = 10
    ENEMY_KILL_REWARD = 5
    BOSS_DAMAGE_REWARD = 10
    HP_PENALTY = 5
    DEATH_PENALTY = 100
    MOVEMENT_REWARD = 0.01
    STUCK_PENALTY = 40
    TOTAL_MOVEMENT_REWARD = 0.05
    BOSS_KILL_REWARD = 1000

    STUCK_THRESHOLD = 1500
    POS_HISTORY = 1000

    def loadState(self, path):
        with open(path, "rb") as f:
            data = f.read()

        state = np.frombuffer(data, dtype=np.uint8).copy()
        self.nes.load(state)
        self.updateState()

    def __init__(self):
        self.nes = WindowedNES("rom.nes")
        #self.nes = NES("rom.nes")
        #self.position_history = deque(maxlen=800)
        self.position_history = deque(maxlen=Env.POS_HISTORY)
        self.stepCount = 0
        self.loadState(getRandomState())
        self.frameBuffer = np.zeros((4,84,84), dtype=np.uint8) #holds stacked four frames
        self.reset()

    def updateState(self):
        self.screenid = self.nes[Env.SCREENID]
        self.hp = self.nes[Env.HP]
        self.enemytable = []
        for i in range(Env.ENEMIES_START, Env.ENEMIES_END + 1):
            self.enemytable.append(self.nes[i])
        self.boss_hp = self.nes[Env.BOSS_HP]
        self.x_pos = self.nes[Env.X_POS]
        self.y_pos = self.nes[Env.Y_POS]
        self.lives = self.nes[Env.LIVES]
        self.position_history.append((self.x_pos, self.y_pos))


    def reset(self):
        self.loadState(getRandomState())

        frame = processSingleFrame(self.getSingleFrame())

        for i in range(4):
            self.frameBuffer[i] = frame

        self.position_history.clear()
        self.stepCount = 0

        return self.getFrameBuffer()

    def step(self, action=0):
        controller_action = Env.action_space[action]
        self.nes.controller = controller_action

        total_reward = 0
        terminal = False
        frame = None

        for i in range(4):
            self.stepCount += 1
            frame = self.nes.step()
            total_reward += self.getReward()
            _, terminal = self.isTerminal()  # capture before updateState clobbers self.lives
            self.updateState()
            if terminal:
                break

        frame = processSingleFrame(frame)
        obs = self.getFrameBuffer()
        self.writeToFrameBuffer(frame)
        next_obs = self.getFrameBuffer()

        return obs, total_reward, next_obs, terminal

    def getReward(self):
        reward = 0

        # screen transition
        if self.nes[Env.SCREENID] != self.screenid:
            self.position_history.clear() #clear position history on room transition
            if self.nes[Env.SCREENID] < self.screenid:
                reward += Env.SCREEN_REWARD
            elif self.nes[Env.SCREENID] > self.screenid:
                reward -= Env.SCREEN_REWARD
        # movement
        reward += (abs(self.x_pos - self.nes[Env.X_POS]) + abs(self.y_pos - self.nes[Env.Y_POS])) * Env.MOVEMENT_REWARD

        # damage taken
        reward -= (self.hp - self.nes[Env.HP]) * Env.HP_PENALTY

        # boss damage
        reward += (self.boss_hp - self.nes[Env.BOSS_HP]) * Env.BOSS_DAMAGE_REWARD

        # enemy kills
        for i in range(0, Env.ENEMIES_END - Env.ENEMIES_START + 1):
            if self.enemytable[i] > self.nes[Env.ENEMIES_START + i]:
                reward += 3

        # death
        terminal_reward, terminal = self.isTerminal()
        if terminal:
            reward += self.getTotalMovement() * Env.TOTAL_MOVEMENT_REWARD
            reward += terminal_reward

        return reward

    #helpers
    def getSingleFrame(self):
        return self.nes.step()

    def writeToFrameBuffer(self, frame):
        self.frameBuffer[:-1] = self.frameBuffer[1:]
        self.frameBuffer[-1] = frame

    def getFrameBuffer(self):
        # we make a deep copy for simplicity of handling on agent side
        return self.frameBuffer.copy().astype(np.float32) / 255.0

    def checkStuck(self):

        if len(self.position_history) == self.position_history.maxlen and self.boss_hp == 0:
            print("checking if stuck")
            if self.getTotalMovement() < Env.STUCK_THRESHOLD:
                print("STUCK!")
                return True

        return False

    """
    def checkStuck(self):
        # only check every  frames

        if self.stepCount >= Env.POS_HISTORY:
            self.stepCount = 0
            if self.getTotalMovement() < Env.STUCK_THRESHOLD:
                return True

        return False
        """

    """def getTotalMovement(self):
        total_movement = sum(
            abs(self.position_history[i][0] - self.position_history[i - 1][0]) +
            abs(self.position_history[i][1] - self.position_history[i - 1][1])
            for i in range(1, len(self.position_history))
        )
        print(f"Total Movement: {total_movement}")
        return total_movement"""

    def getTotalMovement(self):
        displacement = self.displacement(self.position_history[0], self.position_history[-1])
        print(f"Net displacement: {displacement}")
        return displacement

    def displacement(self, v1, v2):
        res = 0
        for i in range(len(v1)):
            compn = v1[i] + v2[i]
            res += compn*compn

        res = math.sqrt(res)
        return res

    def isTerminal(self):
        hp_death = self.nes[Env.HP] == 0
        pit_death = self.nes[Env.LIVES] < self.lives
        stuck_terminal = self.checkStuck()
        boss_killed = self.killedBoss()

        terminal = hp_death or pit_death or stuck_terminal or boss_killed
        reward = Env.BOSS_KILL_REWARD if boss_killed else -Env.DEATH_PENALTY
        if terminal:
            return reward, terminal
        return 0, terminal

    def killedBoss(self):
        return self.boss_hp > 0 and self.nes[Env.BOSS_HP] <= 0
    """
    At time t:
        - You have frames: 1,2,3,4
        - You choose action for frame 4
    At time t+1:
        - You now have frames: 2,3,4,5
        - You choose action for frame 5
    """






