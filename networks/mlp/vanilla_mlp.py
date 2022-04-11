import torch
import torch.nn as nn

from functools import reduce


class VanillaMLP(nn.Module):

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super(VanillaMLP, self).__init__()

        input_size = reduce(lambda x, y: x*y, input_shape)
        output_size = reduce(lambda x, y: x*y, output_shape)

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))

        return x
