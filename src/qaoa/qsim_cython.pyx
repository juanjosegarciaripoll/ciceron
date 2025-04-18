cimport libc.math as cmath
import numpy as np
cimport numpy as cnp
ctypedef cnp.complex128_t complex_t
from libcpp cimport bool

# **********************************
# QUANTUM COMPUTER SIMULATOR
# **********************************

# Definition of required gate
σz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
σx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
σy = -1j * σz @ σx
i2 = np.eye(2, dtype=np.complex128)


cdef void local_product(complex_t *O, complex_t *state, int leading, int rest) noexcept nogil:
    # Implement numpy.einsum('ij,kjl->kil', O, state)
    # where state(leading, 2, rest) and O(2,2)
    # Because we multiply in C ordering,
    # O(ij) * A(jl) = C(il)
    # is actually seen as
    # C^t(li) = A^t(lj) * O^t(ji)
    cdef:
        complex_t a, b
        complex_t O00 = O[0], O01 = O[1], O10 = O[2], O11 = O[3]
    for _ in range(leading):
        for _ in range(rest):
            a = state[0]
            b = state[rest]
            state[0] = O00 * a + O01 * b
            state[rest] = O10 * a + O11 * b
            state += 1
        state += rest


cdef void accumulate_local_product(complex_t *output, complex_t *O, complex_t *state, int leading, int rest) noexcept nogil:
    # Implement numpy.einsum('ij,kjl->kil', O, state)
    # where state(leading, 2, rest) and O(2,2)
    cdef:
        complex_t a, b
        complex_t O00 = O[0], O01 = O[1], O10 = O[2], O11 = O[3]
    for _ in range(leading):
        for _ in range(rest):
            a = state[0]
            b = state[rest]
            output[0] += O00 * a + O01 * b
            output[rest] += O10 * a + O11 * b
            state += 1
            output += 1
        state += rest
        output += rest


def apply_op(cnp.ndarray O, cnp.ndarray ψ):
    cdef:
        cnp.ndarray output = np.empty_like(ψ)
        complex_t *output_state = <complex_t *>output.data
        complex_t *Op = <complex_t *>O.data
        complex_t *state = <complex_t *>ψ.data
        int d = ψ.shape[ψ.ndim-1]
        int l = ψ.size / d
        int N = int(cmath.round(cmath.log2(d)))
    for _ in range(N):
        d >>= 1
        local_product(Op, state, l, d)
        l <<= 1
    return ψ


def apply_sum_op(cnp.ndarray O, cnp.ndarray ψ):
    cdef:
        complex_t *Op = <complex_t *>O.data
        complex_t *state = <complex_t *>ψ.data
        cnp.ndarray output = np.zeros_like(ψ, dtype=np.complex128)
        complex_t *output_state = <complex_t *>output.data
        int d = ψ.shape[ψ.ndim-1]
        int l = ψ.size / d
        int N = int(cmath.round(cmath.log2(d)))
    for _ in range(N):
        d >>= 1
        accumulate_local_product(output_state, Op, state, l, d)
        l <<= 1
    return output


def apply_expiH(double γ, cnp.ndarray E, cnp.ndarray ψ):
    """return np.exp((1j * γ) * E) * ψ"""
    cdef:
        int d = ψ.shape[ψ.ndim-1]
        int l = ψ.size / d
        double *Edata = <double *>E.data
        double *Edata_j
        complex_t *ψdata = <complex_t *>ψ.data
        double angle
    for _ in range(l):
        Edata_j = Edata
        for _ in range(d):
            angle = γ * Edata_j[0]
            ψdata[0] *= cmath.cos(angle) + 1j * cmath.sin(angle)
            Edata_j += 1
            ψdata += 1
    return ψ


Ryderiv = -0.5j * σy
def Ry(θ):
    return np.cos(θ / 2) * i2 - 1j * np.sin(θ / 2) * σy


Rxderiv = -0.5j * σx
def Rx(θ):
    return np.cos(θ / 2) * i2 - 1j * np.sin(θ / 2) * σx


def expH(γ, E):
    return np.exp((1j * γ) * E)


def U1(λ):
    return np.cos(λ / 2) * i2 - 1j * np.sin(λ / 2) * σz
