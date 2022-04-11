import torch.multiprocessing as mp
import torch

from functools import reduce
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from core.utils.agent import make_agent
from envs.make_envs.make_envs import make_parallel_envs

from buffers.utils import merge, wrapper_kv, unwrapper_kv


def main(control_dict, agent_dict, env_dict):
    logfile_name = env_params['env_1']['env_name'] + "_" + \
                   control_params['alg_config_path'].rsplit("/", 1)[-1].split(".")[0] + \
                   datetime.now().strftime("%Y%m%d%H%M%S") + "_train"
    writer = SummaryWriter(log_dir='../logs/{}'.format(logfile_name))
    # 1.创建环境
    envs = make_parallel_envs(env_params)
    # # 2.创建Agent
    agent = make_agent(control_params['alg_config_path'].rsplit("/", 1)[-1].split(".")[0],
                       agent_dict)

    mq = mp.Queue(10000)
    eval_process = mp.Process(target=eval, args=(env_dict['env_1'], mq, agent_dict, env_dict, control_dict))
    eval_process.start()

    mq.put((0, agent.target_policy.state_dict()))

    episodes = control_dict['episodes']

    for episode in range(1, episodes+1):
        obs = envs.reset()
        masks = {env_id: False for env_id in env_dict.keys()}
        envs_id = [env_id for env_id, v in masks.items() if v is False]
        max_train_reward = 0.

        while True:
            actions = wrapper_kv(envs_id, agent.act(unwrapper_kv(obs)))
            data_dicts = envs.step(envs_id, actions)
            agent.memory(merge(obs, actions, data_dicts))
            envs_id = [key for key, value in data_dicts.items() if value[2] is False]

            if not envs_id:
                break

            max_train_reward += 1
            obs = {key: data_dicts[key][1] for key in envs_id}

        update_count, eps = agent.update()
        mq.put((episode, agent.target_policy.state_dict()))
        writer.add_scalar("Train/max return", max_train_reward, episode)
        writer.add_scalar("Train/update_count", update_count, episode)
        writer.add_scalar("Train/eps", eps, episode)

    envs.close()
    mq.put(("end", None))
    eval_process.join()
    eval_process.close()


def eval(env_info, queue, agent_dict, env_params, control_params):
    from envs.make_envs.make_envs import make_env
    from core.utils.networks import instantiate_network
    import random

    logfile_name = env_params['env_1']['env_name'] + "_" + \
                   control_params['alg_config_path'].rsplit("/", 1)[-1].split(".")[0] + \
                   datetime.now().strftime("%Y%m%d%H%M%S") + "_eval"
    writer = SummaryWriter(log_dir='../logs/{}'.format(logfile_name))

    device = torch.device("cuda:{}".format(control_params['eval_device']) if torch.cuda.is_available() else "cpu")

    eps = 0.05

    env = make_env(env_info)

    agent = instantiate_network(agent_dict['network_name'], **{"input_shape": agent_dict['observation_shape'],
                                                               "output_shape": agent_dict['output_shape']})
    agent.eval()
    output_size = reduce(lambda x, y: x*y, agent_dict['output_shape'])

    while True:
        episode, params = queue.get()
        if episode == "end":
            break
        agent.load_state_dict(params)
        ob = env.reset()
        r = 0.
        while True:
            ob = torch.from_numpy(ob).unsqueeze(0)
            env.render()
            if random.random() > eps:
                action = torch.argmax(agent(ob), dim=-1).item()
            else:
                action = random.choice(list(range(output_size)))
            next_ob, reward, done, info = env.step(action)
            r += reward
            if done:
                writer.add_scalar('Eval/return', r, episode)
                break

            ob = next_ob
    # TODO: save the model.
    env.close()


if __name__ == '__main__':
    import yaml

    with open("../configs/control.yml", "r") as f:
        control_params = yaml.load(f, Loader=yaml.FullLoader)

    with open(control_params['alg_config_path'], "r") as f:
        agent_params = yaml.load(f, Loader=yaml.FullLoader)

    with open(control_params['env_config_path'], "r") as f:
        env_params = yaml.load(f, Loader=yaml.FullLoader)

    # 赋值
    agent_params['observation_shape'] = env_params['env_1']['observation_shape']
    agent_params['output_shape'] = env_params['env_1']['output_shape']
    agent_params['action_shape'] = env_params['env_1']['action_shape']
    agent_params['train_device'] = control_params['train_device']
    agent_params['envs_id'] = [key for key in env_params.keys()]

    mp.set_start_method("spawn")

    main(control_params, agent_params, env_params)

