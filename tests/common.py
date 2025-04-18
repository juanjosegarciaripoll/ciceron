import unittest
import numpy as np


def normalize(state):
    return np.array(state, dtype=np.complex128) / np.linalg.norm(state)


class CommonTestCase(unittest.TestCase):
    rng = np.random.default_rng(seed=0x123122)

    def one_qubit_states(self):
        states = [[0, 1], [1, 0], [0.34, 5], [0.5, 0.5]]
        return [normalize(s) for s in states]

    def two_qubit_states(self):
        return [
            np.kron(s1, s2)
            for s1 in self.one_qubit_states()
            for s2 in self.one_qubit_states()
        ]

    def three_qubit_states(self):
        return [
            np.kron(s1, s2)
            for s1 in self.one_qubit_states()
            for s2 in self.two_qubit_states()
        ]

    def assertClose(self, a, b):
        if (a.ndim != b.ndim) or (a.shape != b.shape) or not np.all(np.isclose(a, b)):
            raise AssertionError(f"Unequal tensors:\na = {a}\nb = {b}")
