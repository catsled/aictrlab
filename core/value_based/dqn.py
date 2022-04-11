"""
实现 Deep Q-Learning, epsilon-greedy 版本
"""
import torch
import random
import numpy as np

from functools import reduce
from core.utils.buffers import instantiate_buffer
from core.utils.networks import instantiate_network
from torch.optim.lr_scheduler import ExponentialLR

from torch.optim import Adam


class DQN(object):

    def __init__(self, *args, **kwargs):
        """
        注意：kwargs中包含了所有的超参数
        DQN中必须的参数有：
        lr(learning rate)[double]： 学习率
        eps [float]: 探索概率
        optimizer [str]: 这里需要通过其他函数来实例化一个优化器
        network [nn.Module]: 传递的神经网络名称， 需要通过其他函数进行实例化
        update_interval [int]: 更新target_policy的间隔
        buffer_type [str]: 创建replay_buffer，需要类型，大小等
        minibatch [int]: 更新时batch的大小
        """
        super(DQN, self).__init__()
        self.lr = float(kwargs.get("lr", 1e-4))
        self.eps = float(kwargs.get("eps", 1.))

        device = kwargs.get("device", "0")
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")

        # 创建网络
        self.observation_shape = kwargs.get("observation_shape", (None, ))
        # 这里对应的是应该输出多少个动作，但是在DQN中，其对应的是每个动作对应的Q值
        # 注意不要与其他学习方法混淆，如：gradient-based方法
        self.output_shape = kwargs.get("output_shape", (None, ))
        # 这里是传递给环境的动作的形状
        self.action_shape = kwargs.get("action_shape", (1, ))
        network_name = kwargs.get("network_name", None)

        self.behavior_policy = instantiate_network(network_name,
                                                   **{"input_shape": self.observation_shape,
                                                      "output_shape": self.output_shape})

        self.target_policy = instantiate_network(network_name,
                                                 **{"input_shape": self.observation_shape,
                                                    "output_shape": self.output_shape})

        self.target_policy.load_state_dict(self.behavior_policy.state_dict())

        # 创建优化器
        self.optimizer = Adam(self.behavior_policy.parameters(), lr=self.lr)

        # 随机采样设置
        self.output_size = reduce(lambda x, y: x*y, self.output_shape)
        self.valuable_actions = list(range(self.output_size))

        # 创建replay buffer
        buffer_type = kwargs.get("buffer_type", "uniform")
        buffer_size = kwargs.get("buffer_size", 0)
        envs_id = kwargs.get("envs_id", None)
        self.buffer = instantiate_buffer(buffer_type, envs_id, buffer_size)

        # 其他参数
        self.batch_size = kwargs.get("batch_size", 0)
        self.gamma = kwargs.get("gamma", 0.99)
        self.update_epoch = kwargs.get("update_epoch", 10)
        self.update_interval = int(kwargs.get("update_interval", 5))
        self.update_count = 0
        self.eps_decay_rate = float(kwargs.get("eps_decay_rate", 1e-4))
        self.mse_loss = torch.nn.MSELoss()

        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.9999)

    def act(self, obs):
        """
        这里的obs是指一批observation.
        """
        if random.random() < self.eps:  # 随机探索
            batch = obs.size(0)
            actions = [random.choice(self.valuable_actions) for _ in range(batch)]
        else:  # 贪婪策略
            batch_q_values = self.behavior_policy(obs)
            batch_action_values, actions = torch.max(batch_q_values, 1)
        return torch.from_numpy(np.array(actions))

    def update(self):
        sampler = self.buffer.sample(self.batch_size)
        if sampler is None:
            return self.update_count, self.eps
        for sample in sampler:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = sample

            with torch.no_grad():
                target_y = batch_rewards + self.gamma * \
                           (1. - batch_dones.long()) * (self.target_policy(batch_next_obs).max(1)[0])

            predict_y = self.behavior_policy(batch_obs).gather(1, batch_actions.view(-1, 1).long())

            loss = self.mse_loss(predict_y.view(-1, 1), target_y.view(-1, 1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.update_count += 1

            if self.update_count % self.update_interval == 0:
                self.target_policy.load_state_dict(self.behavior_policy.state_dict())
            self.eps = max(0.1, self.eps - self.eps_decay_rate)

        self.lr_scheduler.step()
        return self.update_count, self.eps

    def memory(self, key_values):
        self.buffer.push(key_values)

    def save(self, path):
        torch.save(self.target_policy.state_dict(), path)

