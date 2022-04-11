"""
简单的CNN实现：参考VGG
"""

import torch
import torch.nn as nn

from functools import reduce


class VanillaCNN(nn.Module):

    def __init__(self, input_shape, output_shape, *args, **kwargs):
        super(VanillaCNN, self).__init__()
        c, h, w = input_shape
        output_size = reduce(lambda x, y: x*y, output_shape)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),  # 32 x h x w
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 x h // 2 x w // 2
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 x h // 2 x w // 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64 x h // 4 x w /// 4
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 x h//4 x w//4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128 x h//8 x w//8
        )

        fh, fw = h//8, w//8

        self.fc1 = nn.Sequential(
            nn.Linear(128*fh*fw, 32*fh*fw),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(32*fh*fw, fh*fw),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(fh*fw, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)

        x = self.fc3(self.fc2(self.fc1(x)))

        return x


# if __name__ == '__main__':
#     x = torch.ones((10, 3, 128, 128))
#     b, c, h, w = x.shape
#
#     output_shape = (1, )
#
#     m = VanillaCNN((c, h, w), output_shape)
#
#     y = m(x)
#
#     print(y)

