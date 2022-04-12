"""
根据网络名称选取对网络进行初始化.
"""
from networks.cnn2d.vanilla_cnn import VanillaCNN
from networks.mlp.vanilla_mlp import VanillaMLP
from networks.mlp.vanilla_mlp_ddpg import VMActor, VMCritic


def instantiate_network(network_name: str, *args, **kwargs):
    if network_name == "vanilla_cnn":
        input_shape = kwargs.get("input_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VanillaCNN(input_shape, output_shape)
    elif network_name == "vanilla_mlp":
        input_shape = kwargs.get("input_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VanillaMLP(input_shape, output_shape)
    elif network_name == "vanilla_mlp_actor":
        input_shape = kwargs.get("input_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VMActor(input_shape, output_shape, **{"action_range": kwargs.get("action_range", 1)})
    elif network_name == "vanilla_mlp_critic":
        observation_shape = kwargs.get("observation_shape", None)
        action_shape = kwargs.get("action_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VMCritic(observation_shape, action_shape, output_shape)
    else:
        raise NotImplementedError

    return network
