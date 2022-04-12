"""
实现 DDPG
"""
import torch
import random
import numpy as np

from functools import reduce
from torch.distributions import Normal
from core.utils.buffers import instantiate_buffer
from core.utils.networks import instantiate_network
from torch.optim.lr_scheduler import ExponentialLR

from torch.optim import Adam


class DDPG(object):
    def __init__(self, *args, **kwargs):
        """
        注意：kwargs中包含了所有的超参数
        DQN中必须的参数有：
        lr(learning rate)[double]： 学习率
        std [float]: 探索概率
        optimizer [str]: 这里需要通过其他函数来实例化一个优化器
        network [nn.Module]: 传递的神经网络名称， 需要通过其他函数进行实例化
        update_interval [int]: 更新target_policy的间隔
        buffer_type [str]: 创建replay_buffer，需要类型，大小等
        batch_size [int]: 更新时batch的大小
        """
        super(DDPG, self).__init__()
        self.lr_actor = float(kwargs.get("lr_actor", 1e-4))
        self.lr_critic = float(kwargs.get("lr_critic", 1e-4))
        self.std = float(kwargs.get("std", 1.))
        self.mean = float(kwargs.get("mean", 0.))
        self.tau = float(kwargs.get("tau", 1e-5))

        device = kwargs.get("device", "0")
        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else "cpu")

        # 创建网络
        self.observation_shape = kwargs.get("observation_shape", (None, None))
        # 这里是传递给环境的动作的形状
        self.action_shape = kwargs.get("action_shape", (1, 1))

        network_name_actor = kwargs.get("network_name_actor", None)
        network_name_critic = kwargs.get("network_name_critic", None)

        self.behavior_actor = instantiate_network(network_name_actor,
                                                  **{"input_shape": self.observation_shape,
                                                      "output_shape": self.action_shape})

        self.target_actor = instantiate_network(network_name_actor,
                                                **{"input_shape": self.observation_shape,
                                                    "output_shape": self.action_shape})

        self.target_actor.load_state_dict(self.behavior_actor.state_dict())

        self.behavior_critic = instantiate_network(network_name_critic,
                                                   **{"observation_shape": self.observation_shape,
                                                      "action_shape": self.action_shape,
                                                      "output_shape": (1, )})

        self.target_critic = instantiate_network(network_name_critic,
                                                 **{"observation_shape": self.observation_shape,
                                                    "action_shape": self.action_shape,
                                                    "output_shape": (1, )})

        self.target_critic.load_state_dict(self.behavior_critic.state_dict())

        # 创建优化器
        self.optimizer_actor = Adam(self.behavior_actor.parameters(), lr=self.lr_actor, weight_decay=1e-2)
        self.optimizer_critic = Adam(self.behavior_critic.parameters(), lr=self.lr_critic)

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
        self.std_decay_rate = float(kwargs.get("std_decay_rate", 1e-4))
        self.mse_loss = torch.nn.MSELoss()

        self.lr_scheduler_actor = ExponentialLR(self.optimizer_actor, gamma=0.9999)
        self.lr_scheduler_critic = ExponentialLR(self.optimizer_critic, gamma=0.9999)

    def act(self, obs, eval=False):
        actions = self.behavior_actor(obs)
        if not eval:
            std = torch.full_like(actions[0], self.std)
            mean = torch.full_like(actions[0], self.mean)
            dist = Normal(loc=mean, scale=std)
            noise = dist.sample()
            actions = actions + noise
        return actions

    def update(self):
        sampler = self.buffer.sample(self.batch_size)
        if sampler is None:
            return self.update_count, self.std
        for sample in sampler:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = sample

            with torch.no_grad():
                batch_actions_prime = self.target_actor(batch_next_obs)
                target_y = batch_rewards + self.gamma * (1 - batch_dones.long()) * \
                           self.target_critic(batch_next_obs, batch_actions_prime).view(-1)

            predict_y = self.behavior_critic(batch_obs, batch_actions)

            critic_loss = self.mse_loss(target_y.view(-1, 1), predict_y)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            actions = self.act(batch_obs, True)
            actor_loss = -self.behavior_critic(batch_obs, actions).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            self.update_count += 1

            if self.update_count % self.update_interval == 0:
                self.target_actor.load_state_dict(self.behavior_actor.state_dict())
                self.target_critic.load_state_dict(self.behavior_critic.state_dict())

            self.std = max(0.1, self.std - self.std_decay_rate)

        return self.update_count, self.std

    def memory(self, key_values):
        self.buffer.push(key_values)


