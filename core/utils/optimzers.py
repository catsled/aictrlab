"""
根据字符串返回相应的优化器实例
"""
from torch.optim import Adam, RMSprop, SGD
import torch


def instantiate_optimizer(optim: str, lr: float, network: torch.nn.Module, *args, **kwargs):
    """
    实例化优化器
    TODO: 需要研究一下每个优化的区别，以及应该传入什么参数，目前只传递学习率
    :param optim:
    :param lr:
    :param network
    :param args:
    :param kwargs:
    :return:
    """
    if optim == "Adam":
        optimizer = Adam(network.parameters(), lr)
    elif optim == "RMSprop":
        optimizer = RMSprop(network.parameters(), lr)
    elif optim == "SGD":
        optimizer = SGD(network.parameters(), lr)
    else:
        raise NotImplementedError

    return optimizer
