from replay_buffer import ReplayBuffer
from model import Model, soft_update
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2


class Agent():

    def __init__(self, env, hidden_layer, learning_rate, step_repeat, gamma):

        self.env = env

        self.step_repeat = step_repeat

        self.gamma = gamma

        obs, info = self.env.reset()

        # TODO - Process  Observation

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print(f'Loaded model on device {self.device}')

        self.memory = ReplayBuffer(max_size = 500000, input_shape = obs.shape, device = self.device)

        self.model = Model(action_dim = env.action_space.n, hidden_dim = hidden_layer, observation_shape = obs.shape).to(self.device)

        self.target_model = Model(action_dim = env.action_space.n, hidden_dim = hidden_layer, observation_shape = obs.shape).to(self.device)

        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

        self.learning_rate = learning_rate


    # 1:37:07 Problems to fix

        