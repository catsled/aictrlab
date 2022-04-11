import torch
import numpy as np

from collections import OrderedDict


def merge(obs_dict: dict, actions_dict:dict, others_dict:dict):
    return_dict = {}
    for key in obs_dict.keys():
        return_dict[key] = []
        return_dict[key].append(obs_dict[key])
        return_dict[key].append(actions_dict[key])
        return_dict[key].extend(others_dict[key][:-1])

    return return_dict


def wrapper_kv(keys: list, values):
    return_dict = {}
    for ind, key in enumerate(keys):
        return_dict[key] = values[ind].numpy()

    return return_dict


def unwrapper_kv(key_values: dict):
    return_values = []
    for key, value in OrderedDict(**key_values).items():
        return_values.append(value)

    return torch.from_numpy(np.array(return_values))
