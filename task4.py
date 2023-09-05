"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, set_pulled, reward): This method is called
        just after the give_pull method. The method should update the
        algorithm's internal state based on the arm that was pulled and the
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull
        but set_pulled is the set that is randomly chosen when the pull is
        requested from the bandit instance.)
"""

import numpy as np


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
        self.num_sets = 2
        self.successes_p_1 = [None] * self.num_sets
        self.failures_p_1 = [None] * self.num_sets
        for i in range(self.num_sets):
            self.successes_p_1[i] = np.ones(num_arms)
            self.failures_p_1[i] = np.ones(num_arms)

    def give_pull(self):
        p_expected = np.zeros(self.num_arms)
        for i in range(self.num_sets):
            p_expected += np.random.beta(self.successes_p_1[i], self.failures_p_1[i])
        p_expected /= self.num_sets
        return np.argmax(p_expected)

    def get_reward(self, arm_index, set_pulled, reward):
        self.successes_p_1[set_pulled][arm_index] += reward
        self.failures_p_1[set_pulled][arm_index] += 1 - reward
