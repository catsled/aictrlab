"""
最简单的replay buffer,采样时完全随机采样。
TODO: 目前不知道其他算法的buffer是怎么实现，先这样实现，后面再优化
"""
from buffers.base_buffer import BaseBuffer


class UniformBuffer(BaseBuffer):

    def __init__(self, keys, max_size):
        super(UniformBuffer, self).__init__(keys, max_size)
