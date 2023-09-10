import numpy as np
from multiprocessing import Pool
from task1 import Eps_Greedy, UCB, KL_UCB, KL
import matplotlib.pyplot as plt


class BernoulliArmTask2:
    def __init__(self, p):
        self.p = p

    def pull(self, num_pulls=None):
        return np.random.binomial(1, self.p, num_pulls)


class BernoulliBanditTask2:
    def __init__(
        self,
        probs=[0.3, 0.5, 0.7],
    ):
        self.__arms = [BernoulliArmTask2(p) for p in probs]
        self.__max_p = max(probs)
        self.__regret = 0

    def pull(self, index):
        reward = self.__arms[index].pull()
        self.__regret += self.__max_p - reward
        return reward

    def regret(self):
        return self.__regret

    def num_arms(self):
        return len(self.__arms)


def single_sim_task2(seed=0, ALGO=Eps_Greedy, PROBS=[0.3, 0.5, 0.7], HORIZON=1000):
    np.random.seed(seed)
    np.random.shuffle(PROBS)
    bandit = BernoulliBanditTask2(probs=PROBS)
    algo_inst = ALGO(num_arms=len(PROBS), horizon=HORIZON)
    for t in range(HORIZON):
        arm_to_be_pulled = algo_inst.give_pull()
        reward = bandit.pull(arm_to_be_pulled)
        algo_inst.get_reward(arm_index=arm_to_be_pulled, reward=reward)
    return bandit.regret()


def simulate_task2(algorithm, probs, horizon, num_sims=50):
    """simulates algorithm of class Algorithm
    for BernoulliBandit bandit, with horizon=horizon
    """

    def multiple_sims(num_sims=50):
        with Pool(10) as pool:
            sim_out = pool.starmap(
                single_sim_task2,
                [(i, algorithm, probs, horizon) for i in range(num_sims)],
            )
        return sim_out

    sim_out = multiple_sims(num_sims)
    regrets = np.mean(sim_out)

    return regrets


def task2(algorithm, horizon, p1s, p2s, num_sims=50):
    """generates the data for task2"""
    probs = [[p1s[i], p2s[i]] for i in range(len(p1s))]

    regrets = []
    for prob in probs:
        regrets.append(simulate_task2(algorithm, prob, horizon, num_sims))

    return regrets


if __name__ == "__main__":
    # part A
    task2p1s = [0.9 for i in range(19)]
    task2p2s = [0.05 * i for i in range(19)]
    regrets = task2(UCB, 30000, task2p1s, task2p2s)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$p_2$")
    ax.set_ylabel(r"$R_T$")
    ax.plot(task2p2s, regrets, color="blue", label="UCB")
    ax.legend()
    ax.set_title(r"Two-armed bandit with $p_1=0.9$ and $T=30000$")
    fig.savefig("task2A.svg", format="svg")

    # part B
    T = 30000
    task2p2s = np.array([0.05 * i for i in range(19)])
    task2p1s = 0.1 + task2p2s
    regrets_UCB = task2(UCB, T, task2p1s, task2p2s)
    regrets_KL_UCB = task2(KL_UCB, T, task2p1s, task2p2s)
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$p_2$")
    ax.set_ylabel(r"$R_T$")
    ax.plot(task2p2s, regrets_UCB, color="blue", label="UCB")
    ax.plot(task2p2s, regrets_KL_UCB, color="red", label="KL-UCB")
    ax.legend()
    ax.set_title(r"Two-armed bandit with $p_1 = p_2 + 0.1$ and $T={}$".format(T))
    fig.savefig("task2B.svg", format="svg")
