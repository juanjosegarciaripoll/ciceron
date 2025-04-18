from qaoa.qsim import σx, σy, σz
import numpy as np
from .common import *


class TestSigmas(CommonTestCase):
    def test_pauli_products(self):
        self.assertClose(σx @ σy, 1j * σz)
        self.assertClose(σy @ σz, 1j * σx)
        self.assertClose(σz @ σx, 1j * σy)
        self.assertClose(σx @ σx, np.eye(2))
        self.assertClose(σy @ σy, np.eye(2))
        self.assertClose(σz @ σz, np.eye(2))

    def test_pauli_matrix_type(self):
        self.assertTrue(isinstance(σx, np.ndarray))
        self.assertTrue(isinstance(σy, np.ndarray))
        self.assertTrue(isinstance(σz, np.ndarray))
