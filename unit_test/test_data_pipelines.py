from envs.make_envs.make_envs import make_parallel_envs, make_env
from collections import OrderedDict
from buffers.uniform_buffer import UniformBuffer


if __name__ == '__main__':
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
        # "env_3": {
        #     "env_name": "CartPole-v1",
        #     "random_seed": 2,
        #     "observation_shape": (4, ),
        #     "action_shape": (1, ),
        #     "output_shape": (2, )
        # }
    }

    envs = make_parallel_envs(envs_info)

    buffer = UniformBuffer(10, (4, ), (1, ))

    obs = envs.reset()

    masks = {env_id: False for env_id in OrderedDict(**envs_info).keys()}
    envs_id = [env_id for env_id, v in masks.items() if v is False]

    while True:
        actions = envs.sample(envs_id)
        next_obs, rewards, dones, infos = envs.step(envs_id, actions)
        buffer.push(obs, actions, rewards, next_obs, dones)
        print()
        [masks.update({envs_id[ind]: done.item()}) for ind, done in enumerate(dones)]
        envs_id = [env_id for env_id, v in masks.items() if v is False]

        if not envs_id:
            obs = envs.reset()
            masks = {env_id: False for env_id in OrderedDict(**envs_info).keys()}
            envs_id = [env_id for env_id, v in masks.items() if v is False]
