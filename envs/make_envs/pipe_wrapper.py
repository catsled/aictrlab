"""
打包环境
"""
import torch
import numpy as np

from collections import OrderedDict


class PipeWrapper(object):

    def __init__(self, local_pipe_list, process_list):
        """
        local_pipe_list: [ {env_id: local_pipe}]
        """
        self.local_pipe_list = local_pipe_list
        self.process_list = process_list

    def reset(self):
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            local_pipe.send(["reset", {env_id: None}])

        data = {}
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            (key, value), = local_pipe.recv().items()
            assert key == env_id
            data.update({key: value})

        return data

    def step(self, envs_id: list, actions: dict):
        # 应该将action处理为可执行的形式
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            if env_id not in envs_id:
                continue
            action = actions[env_id]
            local_pipe.send(["step", {env_id: action}])

        data = {}
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            if env_id not in envs_id:
                continue
            (key, value),  = local_pipe.recv().items()
            assert env_id == key
            data.update({key: value})

        return data

    def sample(self, envs_id: list):
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            if env_id not in envs_id:
                continue
            local_pipe.send(["sample", {env_id: None}])

        data = {}
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            if env_id not in envs_id:
                continue
            (key, value),  = local_pipe.recv().items()
            assert key == env_id
            data.update({key: value})

        return data

    def close(self):
        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            local_pipe.send(["close", {env_id: None}])

        for pipe_info in self.local_pipe_list:
            (env_id, local_pipe),  = pipe_info.items()
            local_pipe.close()
