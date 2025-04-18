# **********************************
# STATE COMPUTATION
# **********************************

# Optimal angles
import scipy.optimize
import numpy as np
from numpy import pi as π
from . import p_layers, p_layers_cython
from typing import Callable, Optional
import numpy.typing as npt

#
# PYTHON VERSION
#


def traditional_QAOA_cost_function(x, E):
    θ, γ = x.reshape(2, -1)
    ψ = p_layers.traditional_QAOA_state(θ, γ, E)
    return np.vdot(ψ, E * ψ).real


def traditional_QAOA_cost_function_with_derivatives(x, E):
    θ, γ = x.reshape(2, -1)
    ψ, dθ, dγ = p_layers.traditional_QAOA_state_and_derivatives(θ, γ, E)
    E, dE = np.vdot(ψ, E * ψ).real, np.array(
        [2 * np.vdot(ψ, E * v).real for v in dθ + dγ]
    )
    return E, dE


def QAOA_cost_function(x, E):
    θ, γ = x.reshape(2, -1)
    ψ = p_layers.QAOA_state(θ, γ, E)
    return np.vdot(ψ, E * ψ).real


def QAOA_cost_function_with_derivatives(x, E):
    θ, γ = x.reshape(2, -1)
    ψ, dθ, dγ = p_layers.QAOA_state_and_derivatives(θ, γ, E)
    return np.vdot(ψ, E * ψ).real, np.array(
        [2 * np.vdot(ψ, E * v).real for v in dθ + dγ]
    )


#
# CYTHON VERSION
#


def traditional_QAOA_cost_function_cython(x, E):
    θ, γ = x.reshape(2, -1)
    ψ = p_layers_cython.traditional_QAOA_state(θ, γ, E)
    return np.vdot(ψ, E * ψ).real


def traditional_QAOA_cost_function_cython_with_derivatives(x, E):
    θ, γ = x.reshape(2, -1)
    ψ, dθ, dγ = p_layers_cython.traditional_QAOA_state_and_derivatives(θ, γ, E)
    E, dE = np.vdot(ψ, E * ψ).real, np.array(
        [2 * np.vdot(ψ, E * v).real for v in dθ + dγ]
    )
    return E, dE


def QAOA_cost_function_cython(x, E):
    θ, γ = x.reshape(2, -1)
    ψ = p_layers_cython.QAOA_state(θ, γ, E)
    return np.vdot(ψ, E * ψ).real


def QAOA_cost_function_cython_with_derivatives(x, E):
    θ, γ = x.reshape(2, -1)
    ψ, dθ, dγ = p_layers_cython.QAOA_state_and_derivatives(θ, γ, E)
    return np.vdot(ψ, E * ψ).real, np.array(
        [2 * np.vdot(ψ, E * v).real for v in dθ + dγ]
    )


def select_qaoa_cost_function(
    use_cython: bool, use_gradient: bool, traditional: bool
) -> Callable:
    if use_cython:
        if use_gradient:
            cost_function = (
                traditional_QAOA_cost_function_cython_with_derivatives
                if traditional
                else QAOA_cost_function_cython_with_derivatives
            )
        else:
            cost_function = (
                traditional_QAOA_cost_function_cython
                if traditional
                else QAOA_cost_function_cython
            )
    else:
        if use_gradient:
            cost_function = (
                traditional_QAOA_cost_function_with_derivatives
                if traditional
                else QAOA_cost_function_with_derivatives
            )
        else:
            cost_function = (
                traditional_QAOA_cost_function if traditional else QAOA_cost_function
            )
    return cost_function


def make_initial_point(
    layers: int,
    initial_point: Optional[npt.ArrayLike],
    smooth_interpolation: bool = False,
) -> npt.ArrayLike:
    if initial_point is None:
        θ0 = [(π / 3) / layers + (np.random.rand() - 0.5) * 10 ** (-2)] * layers
        γ0 = [0.1 / layers + (np.random.rand() - 0.5) * 10 ** (-2)] * layers
    else:
        θ0, γ0 = np.reshape(initial_point, (2, -1))
        l = len(θ0)
        if l > 1 and smooth_interpolation:
            x = np.linspace(0, 1, layers)
            x0 = np.linspace(0, 1, len(θ0))
            θ0 = np.interp(x, x0, θ0)
            γ0 = np.interp(x, x0, γ0)
        else:
            θ0 = [np.sum(θ0) / layers] * layers
            γ0 = [np.sum(γ0) / layers] * layers
    return np.concatenate((θ0, γ0))


def solve_qaoa(
    E,
    layers=1,
    initial_point=None,
    method="L-BFGS-B",
    traditional=True,
    use_gradient=True,
    check_gradient=False,
    return_state=False,
    alternative_cost=False,
    use_cython=False,
    smooth_interpolation=False,
    **kwargs,
):
    """Solve a QAOA problem of 'N' qubits with energies 'E' sorted by
    index in the states of the qubit register, minimizing the two angles
    θ and γ in the QAOA ansatz."""
    if alternative_cost:
        cost = np.exp(E)
    else:
        cost = E

    initial_point = make_initial_point(
        layers, initial_point, smooth_interpolation=smooth_interpolation
    )
    # print("start:\n", initial_point)

    ## System size
    LL = np.log(E.size) / np.log(2)

    bounds = [(0, π)] * layers + [(0, 2 * π * LL)] * layers
    cost_function = select_qaoa_cost_function(use_cython, use_gradient, traditional)
    if smooth_interpolation:
        bounds = None

    if use_gradient and check_gradient:
        obj = lambda x: cost_function(x, cost)
        grad = lambda x: cost_function(x, cost)[1]
        deriv = scipy.optimize.approx_fprime(initial_point, obj)
        err = scipy.optimize.check_grad(
            obj,
            grad,
            initial_point,
        )
        print(f"Gradient error = {err}")
        print(f"  estimated: {deriv}")
        print(f"  algorithm: {grad(initial_point)}")

    result = scipy.optimize.minimize(
        cost_function,
        initial_point,
        args=(cost,),
        jac=use_gradient,
        method=method,
        bounds=bounds,
        tol=1e-10,
        **kwargs,
    )

    θ, γ = result.x.reshape(2, -1)
    energy = result.fun
    if return_state:
        if traditional:
            psi = p_layers.traditional_QAOA_state(θ, γ, E)
        else:
            psi = p_layers.QAOA_state(θ, γ, E)
        return θ, γ, energy, result.nit, E, psi
    else:
        return θ, γ, energy, result.nit


# **********************************
# OBSERVABLES
# **********************************
def GS_probability(ψ2, E):
    # We round to avoid floating-point errors
    ndx = np.where(np.round(E, 5) == np.round(np.min(E), 5))
    return np.sum(ψ2[ndx])


def GS_probability_boltzmann(β, E):
    p = np.exp(-β * E)
    p /= np.sum(p)
    return GS_probability(p, E)


def GS_degeneracy(E):
    # We round to avoid floating-point errors
    return len(np.where(np.round(E, 5) == np.round(np.min(E), 5))[0])


# Effective temperature
def BoltzmannFit_Lin(ψ2, E):
    En = np.sort(E)
    indx = np.argsort(E)
    ψ2 = np.log(ψ2[indx])
    popt, pcov = np.polyfit(En, ψ2, 1, cov=True)
    β = popt[0]
    β_std = np.sqrt(pcov[0, 0])
    org = popt[1]
    org_std = np.sqrt(pcov[1, 1])
    return β, β_std, org, org_std


# **********************************
# EXPERIMENTS
# **********************************
class OptimalAnglesExperiment:
    def __init__(
        self,
        E,
        use_cython=False,
        fit=True,
        smooth_interpolation=False,
        layers=1,
        previous=None,
    ):
        self.use_cython = use_cython
        if previous:
            initial_point = np.concatenate((previous.theta, previous.gamma))
        else:
            initial_point = None
        self.layers = layers
        self.theta, self.gamma, self.energy, self.nit, E, self.psi = solve_qaoa(
            E,
            initial_point=initial_point,
            layers=layers,
            use_gradient=True,
            return_state=True,
            use_cython=use_cython,
            smooth_interpolation=smooth_interpolation,
        )
        QAOA_Prob = np.abs(self.psi) ** 2
        self.energy = np.sum(E * QAOA_Prob)
        self.pgs = GS_probability(QAOA_Prob, E)
        # ============== CALCULATE PARTICIPATION ENERGY AND FITS==============================
        self.E_std = np.sqrt(np.sum(E**2 * QAOA_Prob) - self.energy**2)
        self.E_PR = self.energy + (self.layers / 2) * self.E_std

        ndPR = np.where(E < self.E_PR)
        self.psi_1 = QAOA_Prob[ndPR]
        self.Ener_1 = E[ndPR]
        if fit:
            self.beta, self.beta_std, self.Alin, self.Alin_std = BoltzmannFit_Lin(
                QAOA_Prob, E
            )
            if len(self.psi_1) > 2:
                # Fit to lowest energies
                (
                    self.βlin_1,
                    self.βlin_std_1,
                    self.Alin_1,
                    self.Alin_std_1,
                ) = BoltzmannFit_Lin(self.psi_1, self.Ener_1)
            else:
                self.βlin_1, self.βlin_std_1, self.Alin_1, self.Alin_std_1 = [
                    None,
                    None,
                    None,
                    None,
                ]

            ndPR_2 = np.where(E > self.E_PR)
            psi_2 = QAOA_Prob[ndPR_2]
            Ener_2 = E[ndPR_2]
            if len(psi_2) > 2:
                # Fit to highest energies
                (
                    self.βlin_2,
                    self.βlin_std_2,
                    self.Alin_2,
                    self.Alin_std_2,
                ) = BoltzmannFit_Lin(psi_2, Ener_2)
            else:
                self.βlin_2, self.βlin_std_2, self.Alin_2, self.Alin_std_2 = [
                    None,
                    None,
                    None,
                    None,
                ]
        else:
            self.beta = None
            self.βlin_1 = None
            self.βlin_2 = None
