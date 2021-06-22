from typing import Dict, List
import numpy as np
from numpy.linalg import norm
from scipy.special import softmax
from scipy.stats import poisson

"""
Solution to the modified Jack's car rental problem (Exercise 4.7, page 82).
"""

class PolicyIterator:
    def __init__(
            self,
            size: List[int],
            possible_returns: List[object]):
        self.size = np.add(size, 1)
        self.possible_returns = possible_returns

    def new_values(  # TODO
            self,
            values: List[object],
            policy: List[object]
            ) -> List[object]:
        pass

    def new_policy(  # TODO
            self,
            policy: List[object],
            values: List[object]
            ) -> List[object]:
        pass

    def evaluate(
            self,
            values: List[object],
            policy: List[object],
            threshold: float
            ) -> List[object]:
        new_values = values.copy()
        while True:
            old_values = new_values.copy()
            new_values = self.new_values(old_values, policy)
            if norm(new_values - old_values) < threshold:
                return new_values

    def improve(
            self,
            policy: List[object]
            values: List[object],
            threshold: float,
            ) -> List[object], List[object]:
        values_threshold = 1
        new_policy = policy.copy()
        while True:
            values = self.evaluate(values, new_policy, values_threshold)
            old_policy = new_policy.copy()
            new_policy = self.new_policy(old_policy, values)
            if norm(new_policy - old_policy) < threshold:
                return new_policy, values


def main():
    size = [20, 20]
    exp_return_values = [3, 4]
    exp_request_values = [3, 2]
    possible_returns = [[{"value": 0, "probability": 0}]]  # TODO
    values = [[]]  # TODO
    policy = [[[]]]  # TODO
    threshold = 1

    policy_iterator = PolicyIterator(size, possible_returns)
    policy, values = policy_iterator.improve(policy, values, threshold)


if __name__ == "__main__":
    main()
