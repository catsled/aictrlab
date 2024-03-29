from core.value_based.ddqn import DDQN
from core.value_based.dqn import DQN
from core.gradient_based.ddpg import DDPG


def make_agent(agent_name: str, agent_dict):
    if agent_name == "dqn":
        agent = DQN(**agent_dict)
    elif agent_name == "ddqn":
        agent = DDQN(**agent_dict)
    elif agent_name == "ddpg":
        agent = DDPG(**agent_dict)
    else:
        raise NotImplementedError

    return agent
