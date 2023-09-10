"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, reward): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math

# Hint: math.log is much faster than np.log for scalars


class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon

    def give_pull(self):
        raise NotImplementedError

    def get_reward(self, arm_index, reward):
        raise NotImplementedError


# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)

    def get_reward(self, arm_index, reward):
        n = self.counts[arm_index]
        self.values[arm_index] = (n * self.values[arm_index] + reward) / (n + 1)
        self.counts[arm_index] = n + 1


def KL(p, q):
    if p == 0:
        return -math.log(1 - q)
    elif p == 1:
        return -math.log(q)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def KLi(p, z):
    """
    returns q in [p, 1) st
        KL(p, q) = z
    computed using binary search
    """
    cutoff = 1e-3
    r = 1
    l = p
    while (r - l) > cutoff:
        m = (l + r) / 2
        if KL(p, m) > z:
            r = m
        else:
            l = m
    return l


np_KLi = np.vectorize(KLi)


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0

    def give_pull(self):
        if self.time < self.num_arms:
            return self.time
        else:
            return np.argmax(
                self.values + np.sqrt(2 * math.log(self.time) / self.counts)
            )

    def get_reward(self, arm_index, reward):
        n = self.counts[arm_index]
        self.time += 1
        self.values[arm_index] = (n * self.values[arm_index] + reward) / (n + 1)
        self.counts[arm_index] = n + 1


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.time = 0
        self.c = 0

    def give_pull(self):
        if self.time < self.num_arms:
            return self.time
        else:
            l = math.log(self.time)
            l = l + self.c * math.log(l)
            return np.argmax(np_KLi(self.values, l / self.counts))

    def get_reward(self, arm_index, reward):
        n = self.counts[arm_index]
        self.time += 1
        self.values[arm_index] = (n * self.values[arm_index] + reward) / (n + 1)
        self.counts[arm_index] = n + 1


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        self.successes_p_1 = np.ones(num_arms)
        self.failures_p_1 = np.ones(num_arms)

    def give_pull(self):
        return np.argmax(np.random.beta(self.successes_p_1, self.failures_p_1))

    def get_reward(self, arm_index, reward):
        self.successes_p_1[arm_index] += reward
        self.failures_p_1[arm_index] += 1 - reward
