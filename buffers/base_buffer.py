from torch.utils.data import SubsetRandomSampler, BatchSampler

import numpy
import torch
from envs.make_envs.make_envs import make_parallel_envs


class BaseBuffer(object):

    def __init__(self, keys: list, max_size: int, *args, **kwargs):
        self.kv = {
            key: {
                "obs": [],
                "actions": [],
                "rewards": [],
                "next_obs": [],
                "dones": [],
                "count": 0
            }
            for key in keys
        }

        self.max_size = max_size

        self.pool = [None] * self.max_size  # (ob, action, reward, next_ob, done)

        self.pool_count = 0
        self.capacity = 0

    def push(self, key_values: dict):
        """
        key_values: {"env_1": [ob, action, reward, next_ob, done], "env_2":[ob, action, reward, next_ob, done], ...}
        """
        for key, values in key_values.items():
            ob, action, reward, next_ob, done = values
            self.kv[key]['obs'].append(ob)
            self.kv[key]['actions'].append(action)
            self.kv[key]['rewards'].append(reward)
            self.kv[key]['next_obs'].append(next_ob)
            self.kv[key]['dones'].append(done)
            self.kv[key]['count'] += 1
            self.capacity = min(self.capacity+1, self.max_size)

    def serialize(self):
        for key, values in self.kv.items():
            for index in range(values['count']):
                self.pool[self.pool_count] = (
                    values['obs'][index], values['actions'][index], values['rewards'][index],
                    values['next_obs'][index], values['dones'][index]
                )
                self.pool_count = (self.pool_count + 1) % self.max_size
            [self.kv[key][i].clear() for i, _ in values.items() if i != "count"]
            self.kv[key]['count'] = 0

    def sample(self, batch_size):
        if batch_size > self.capacity:
            return None

        self.serialize()

        sampler = BatchSampler(SubsetRandomSampler(list(range(self.capacity))), batch_size=batch_size, drop_last=False)
        for indices in sampler:
            batch_obs = []
            batch_actions = []
            batch_rewards = []
            batch_next_obs = []
            batch_dones = []
            for index in indices:
                ob, action, reward, next_ob, done = self.pool[index]
                batch_obs.append(ob)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_obs.append(next_ob)
                batch_dones.append(done)

            batch_obs = torch.from_numpy(numpy.array(batch_obs))
            batch_actions = torch.from_numpy(numpy.array(batch_actions))
            batch_rewards = torch.from_numpy(numpy.array(batch_rewards)).float()
            batch_next_obs = torch.from_numpy(numpy.array(batch_next_obs))
            batch_dones = torch.from_numpy(numpy.array(batch_dones))

            yield batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


def test():
    from buffers.utils import merge

    envs_info = {
        "env_1": {
            "env_name": "CartPole-v1",
            "random_seed": 0,
            "observation_shape": (4, ),
            "action_shape": (1, ),
            "output_shape": (2, )
        },
        "env_2": {
            "env_name": "CartPole-v1",
            "random_seed": 1,
            "observation_shape": (4, ),
            "action_shape": (1, ),
            "output_shape": (2, )
        },
        "env_3": {
            "env_name": "CartPole-v1",
            "random_seed": 2,
            "observation_shape": (4, ),
            "action_shape": (1, ),
            "output_shape": (2, )
        }
    }

    envs = make_parallel_envs(envs_info)
    masks = {env_id: False for env_id in envs_info.keys()}
    envs_id = [env_id for env_id, v in masks.items() if v is False]

    max_size = 300

    buffer = BaseBuffer(envs_id, max_size)

    obs = envs.reset()

    while True:
        actions = envs.sample(envs_id)
        data_dicts = envs.step(envs_id, actions)
        buffer.push(merge(obs, actions, data_dicts))
        envs_id = [key for key, value in data_dicts.items() if value[2] is False]

        if not envs_id:
            break

        obs = {key: data_dicts[key][1] for key in envs_id}


