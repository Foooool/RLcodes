# coding: utf-8

import numpy as np
from . import TableAgent
from . import SnakeEnv


def eval_game(env, policy):
    """
    Run the environment with specific policy
    Params:
        env     gym.Env
        policy  agent with play function
                list of actions
    Return sum of reward
    """
    state = env.reset()
    return_val = 0
    while True:
        # select an act
        if isinstance(policy, TableAgent):
            act = policy.play(state)
        elif isinstance(policy, list):
            act = policy[state]
        else:
            raise(RuntimeError("Illegal policy"))
        
        # carry out the action
        state, reward, terminate, _ = env.step(act)

        return_val += reward

        if terminate:
            break

    return return_val


def test_easy():
    """
    Test three policies
    Policy opt: throw dice 0 in the first 97 states and dice 1 in the last 3
    Policy   0: throw dice 0 all the times
    Policy   1: throw dice 1 all the times
    """
    # three policies
    policy_ref = [0] * 97 + [1] * 3
    policy_0 = [0] * 100
    policy_1 = [1] * 100

    # record rewards of the three policies
    sum_opt = 0
    sum_0 = 0
    sum_1 = 0

    # prepare the environment
    env = SnakeEnv(0, [3, 6])

    # run each policy 10000 times
    for i in range(10000):
        sum_opt += eval_game(env, policy_ref)
        sum_0 += eval_game(env, policy_0)
        sum_1 += eval_game(env, policy_1)
        
        # show the rate of the process
        if i % 100 == 0:
            print("\t{}% done".format(i / 10000))

    # show the result
    print("Policy opt:", sum_opt/10000)
    print("Policy   0:", sum_0/10000)
    print("Policy   1:", sum_1/10000)


def policy_evaluation(agent, max_iter=-1):
    """
    策略评估步骤
    """
    iteration = 0
    while True:
        # one iteration
        iteration += 1
        new_value_pi = agent.value_pi.copy()
        for i in range(1, agent.s_len):  # for each state
            value_sas = []
            ac = agent.pi[i]
            for j in range(agent.a_len):  # for each act
                transition = agent.p[ac, i, :]
                value_sa = np.dot(transition, agent.r + agent.gamma*agent.value_pi)
                value_sas.append(value_sa)
            new_value_pi[i] = value_sa
        diff = np.sqrt(np.sum(np.power(agent.value_pi-new_value_pi, 2)))
        if diff < 1e-6:
            break
        else:
            agent.value_pi = new_value_pi
        if iteration == max_iter:
            break
