"""
根据网络名称选取对网络进行初始化.
"""
from networks.cnn2d.vanilla_cnn import VanillaCNN
from networks.mlp.vanilla_mlp import VanillaMLP


def instantiate_network(network_name: str, *args, **kwargs):
    if network_name == "vanilla_cnn":
        input_shape = kwargs.get("input_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VanillaCNN(input_shape, output_shape)
    elif network_name == "vanilla_mlp":
        input_shape = kwargs.get("input_shape", None)
        output_shape = kwargs.get("output_shape", None)
        network = VanillaMLP(input_shape, output_shape)
    else:
        raise NotImplementedError

    return network
