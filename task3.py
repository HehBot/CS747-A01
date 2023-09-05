"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.

    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).

    - get_reward(self, arm_index, reward): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np


# START EDITING HERE
# You can use this space to define any helper functions that you need
def beta_trunc(alpha, beta, l, u):
    x = np.random.beta(alpha, beta)
    while x < l or x > u:
        x = np.random.beta(alpha, beta)
    return x


beta_trunc_vectorized = np.vectorize(beta_trunc, excluded=["l", "u"])
# END EDITING HERE


class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault  # probability that the bandit returns a faulty pull
        self.successes_p_1 = np.ones(num_arms)
        self.failures_p_1 = np.ones(num_arms)

    def give_pull(self):
        return np.argmax(
            beta_trunc_vectorized(
                self.successes_p_1,
                self.failures_p_1,
                self.fault / 2,
                1 - (self.fault / 2),
            )
        )

    def get_reward(self, arm_index, reward):
        self.successes_p_1[arm_index] += reward
        self.failures_p_1[arm_index] += 1 - reward
