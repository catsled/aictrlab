"""
创建buffer
"""
from buffers.uniform_buffer import UniformBuffer
from buffers.base_buffer import BaseBuffer


def instantiate_buffer(buffer_type, envs_id, size):
    if buffer_type == "uniform":
        buffer = BaseBuffer(envs_id, size)
    else:
        raise NotImplementedError

    return buffer