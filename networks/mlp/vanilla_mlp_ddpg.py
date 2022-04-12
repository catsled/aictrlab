import torch
import torch.nn as nn

from functools import reduce


class VMActor(nn.Module):

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super(VMActor, self).__init__()

        self.input_size = reduce(lambda x, y: x*y, input_shape)
        self.output_size = reduce(lambda x, y: x*y, output_shape)

        self.action_range = float(kwargs.get("action_range", 1))

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(128, self.output_size)

    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))

        x = torch.tanh(x) * self.action_range

        return x


class VMCritic(nn.Module):

    def __init__(self, observation_shape, action_shape, output_shape, *args, **kwargs):
        super(VMCritic, self).__init__()

        self.observation_size = reduce(lambda x, y: x*y, observation_shape)
        self.action_size = reduce(lambda x, y: x*y, action_shape)
        self.input_size = self.observation_size + self.action_size

        self.output_size = reduce(lambda x, y: x*y, output_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(128, self.output_size)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = self.fc3(self.fc2(self.fc1(x)))

        return x
