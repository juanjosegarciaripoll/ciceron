from ..common import *
import qaoa.qsim
import qaoa.qsim_cython
from qaoa.qsim_cython import σz, σx, σy
import numpy as np


def normalize(state):
    return np.array(state, dtype=np.complex128) / np.linalg.norm(state)


class TestApplyOp(CommonTestCase):
    def invalid_test_cython_apply_op_modifies_in_place(self):
        for state in self.one_qubit_states():
            s = state + 0.0
            new_s = qaoa.qsim_cython.apply_op(σy, s)
            self.assertTrue(np.all(s == new_s))
            self.assertFalse(np.all(s == state))

    def test_cython_apply_op_1_qubit(self):
        for state in self.one_qubit_states():
            self.assertClose(
                qaoa.qsim.apply_op(σz, state + 0.0),
                qaoa.qsim_cython.apply_op(σz, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σx, state + 0.0),
                qaoa.qsim_cython.apply_op(σx, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σy, state + 0.0),
                qaoa.qsim_cython.apply_op(σy, state + 0.0),
            )

    def test_cython_apply_op_2_qubits(self):
        for state in self.two_qubit_states():
            self.assertClose(
                qaoa.qsim.apply_op(σz, state + 0.0),
                qaoa.qsim_cython.apply_op(σz, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σx, state + 0.0),
                qaoa.qsim_cython.apply_op(σx, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σy, state + 0.0),
                qaoa.qsim_cython.apply_op(σy, state + 0.0),
            )

    def test_cython_apply_op_3_qubits(self):
        for state in self.three_qubit_states():
            self.assertClose(
                qaoa.qsim.apply_op(σz, state + 0.0),
                qaoa.qsim_cython.apply_op(σz, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σx, state + 0.0),
                qaoa.qsim_cython.apply_op(σx, state + 0.0),
            )
            self.assertClose(
                qaoa.qsim.apply_op(σy, state + 0.0),
                qaoa.qsim_cython.apply_op(σy, state + 0.0),
            )
