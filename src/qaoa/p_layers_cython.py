import numpy as np
from numpy import pi as π
from .qsim_cython import *
from . import p_layers


# **********************************
# EXTENDED-QAOA SIMULATOR
# **********************************

DEFAULT_LAMBDA = π / 2


def apply_QAOA(ψ, θ, γ, E, λ=π / 2):
    return apply_op(Ry(θ) @ U1(λ), apply_expiH(γ, E, ψ))


def apply_QAOA_derivatives(ψ, θ, γ, E, λ=DEFAULT_LAMBDA):
    ψ = apply_expiH(γ, E, ψ)
    dψdγ = 1j * (E * ψ)
    R = Ry(θ) @ U1(λ)
    dψdγ = apply_op(R, dψdγ)
    ψ = apply_op(R, ψ)
    dψdθ = apply_sum_op(Ryderiv @ U1(λ), ψ)
    return ψ, dψdθ, dψdγ


def make_lambda(γ, λ):
    return [0.0] * (len(γ) - 1) + [λ]
    return [λ] * len(γ)


def QAOA_state(θ, γ, E, λ=DEFAULT_LAMBDA):
    ψ = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    for θi, γi, λi in zip(θ, γ, make_lambda(γ, λ)):
        ψ = apply_QAOA(ψ, θi, γi, E, λi)
    return ψ


def QAOA_state_and_derivatives(θ, γ, E, λ=DEFAULT_LAMBDA):
    ψ = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    dθ = []
    dγ = []
    for θi, γi, λi in zip(θ, γ, make_lambda(γ, λ)):
        ψ, dθi, dγi = apply_QAOA_derivatives(ψ, θi, γi, E, λi)
        dθ = [apply_QAOA(v, θi, γi, E, λi) for v in dθ] + [dθi]
        dγ = [apply_QAOA(v, θi, γi, E, λi) for v in dγ] + [dγi]
    return ψ, dθ, dγ


# **********************************
# TRADITIONAL QAOA
# **********************************


def apply_traditional_QAOA(ψ, θ, γ, E):
    return apply_op(Rx(θ), apply_expiH(γ, E, ψ))


def apply_traditional_QAOA_derivatives(ψ, θ, γ, E):
    ψ = apply_expiH(γ, E, ψ)
    dψdγ = 1j * (E * ψ)
    R = Rx(θ)
    dψdγ = apply_op(R, dψdγ)
    ψ = apply_op(R, ψ)
    dψdθ = apply_sum_op(Rxderiv, ψ)
    return ψ, dψdθ, dψdγ


def traditional_QAOA_state(θ, γ, E):
    ψ = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    for θi, γi in zip(θ, γ):
        ψ = apply_traditional_QAOA(ψ, θi, γi, E)
    return ψ


def traditional_QAOA_state_and_derivatives(θ, γ, E):
    ψ = np.full(E.shape, 1 / np.sqrt(E.size), dtype=np.complex128)
    dθ = []
    dγ = []
    for θi, γi in zip(θ, γ):
        ψ, dθi, dγi = apply_traditional_QAOA_derivatives(ψ, θi, γi, E)
        dθ = [apply_traditional_QAOA(v, θi, γi, E) for v in dθ] + [dθi]
        dγ = [apply_traditional_QAOA(v, θi, γi, E) for v in dγ] + [dγi]
    return ψ, dθ, dγ
