import numpy as np
import math

# **********************************
# QUANTUM COMPUTER SIMULATOR
# **********************************

# Definition of required gate
σz = np.array([[1, 0], [0, -1]])
σx = np.array([[0, 1], [1, 0]])
σy = -1j * σz @ σx
i2 = np.eye(2)


def local_operator(operator, N, i):
    return np.kron(np.eye(2 ** (N - i - 1)), np.kron(operator, np.eye(2**i)))


def product(operators):
    output = 1
    for op in operators:
        output = np.kron(op, output)
    return output


def apply_op(O, ψ):
    N = round(math.log2(ψ.size))
    for _ in range(N):
        ψ = (ψ.reshape(-1, 2) @ O.T).transpose()
    return ψ.flatten()


def apply_sum_op(O, ψ):
    N = round(np.log2(ψ.size))
    Oψ = 0
    for i in range(N):
        Oψ += np.einsum("ij,kjl->kil", O, ψ.reshape(2**i, 2, -1)).flatten()
    return Oψ


def Ry(θ):
    return np.cos(θ / 2) * i2 - 1j * np.sin(θ / 2) * σy


def Rx(θ):
    return np.cos(θ / 2) * i2 - 1j * np.sin(θ / 2) * σx


def apply_expiH(γ, E, ψ):
    return np.exp((1j * γ) * E) * ψ


def expH(γ, E):
    return np.exp((1j * γ) * E)


def U1(λ):
    return np.cos(λ / 2) * i2 - 1j * np.sin(λ / 2) * σz
