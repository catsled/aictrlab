"""
创建worker
"""
from envs.make_envs.make_envs import make_env


def worker(env_comm_info):
    """
    env_comm_info: 环境交互信息
    应该包含环境信息环境{
        {env_info: {}, comm_info: (pipe1, pipe2)}
    }.
    """
    env_id = list(env_comm_info['env_info'].keys())[0]
    env = make_env(env_comm_info['env_info'][env_id])  # 创建环境

    local_pipe, remote_pipe = env_comm_info['comm_info']
    remote_pipe.close()
    num_step = 0

    try:
        while True:
            cmd, data = local_pipe.recv()
            key = list(data.keys())[0]
            data = data[key]
            assert key == env_id

            if cmd == "reset":
                ob = env.reset(seed=env_comm_info['env_info'][key]['random_seed'])
                local_pipe.send({env_id: ob})
            elif cmd == "step":
                next_ob, reward, done, info = env.step(data)
                info['num_step'] = num_step
                local_pipe.send({env_id: [reward, next_ob, done, info]})
                if done:
                    num_step = 0
                else:
                    num_step += 1
            elif cmd == "sample":
                action = env.action_space.sample()
                local_pipe.send({env_id: action})
            elif cmd == "close":
                break
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        local_pipe.close()
        print("{}已关闭".format(env_id))
