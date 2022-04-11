"""
创建并行环境
"""
import gym
import torch.multiprocessing as mp

from envs.make_envs.pipe_wrapper import PipeWrapper


def make_env(env_info: dict):
    # 创建单个环境
    try:
        env = gym.make(env_info['env_name'])
    except gym.error.DeprecatedEnv as e:
        raise NotImplementedError

    return env


def make_parallel_envs(envs_info: dict, *args, **kwargs):
    """
    这里创建时存在这样一个问题：是根据env_info中环境的个数创建对应数量的并行环境，还是根据num_workers的值创建对应数量的环境？
    暂时先根据env_info中环境的个数进行创建
    """
    from envs.make_envs.worker import worker

    pipes = [mp.Pipe() for _ in range(len(envs_info.keys()))]  # 创建管道

    envs_comm_info_list = [
        {"env_info": {key: value}, "comm_info": (pipes[ind][1], pipes[ind][0])} for ind, (key, value) in
        enumerate(envs_info.items())
    ]

    local_pipe_list = []
    process_list = []
    for ind in range(len(pipes)):
        p = mp.Process(target=worker, args=(envs_comm_info_list[ind], ))
        p.start()
        process_list.append(p)
        pipes[ind][1].close()
        env_id = list(envs_comm_info_list[ind]['env_info'].keys())[0]
        local_pipe_list.append({env_id: pipes[ind][0]})

    return PipeWrapper(local_pipe_list, process_list)
