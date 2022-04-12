"""
创建buffer
"""
from buffers.uniform_buffer import UniformBuffer


def instantiate_buffer(buffer_type, envs_id, size):
    if buffer_type == "uniform":
        buffer = UniformBuffer(envs_id, size)
    else:
        raise NotImplementedError

    return buffer
