# **********************************
# GRAPH GENERATION
# **********************************
import networkx as nx
import numpy as np
import numpy.typing as npt
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum
import h5py


@dataclass
class GraphType:
    @abstractmethod
    def make_random_graph(self, vertices: int) -> npt.NDArray:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


@dataclass
class ErdosReny(GraphType):
    density: float

    def __post_init__(self):
        if self.density < 0 or self.density > 1:
            raise Exception(f"Wrong density for an Erdos-Renyi graph {self.density}")

    def make_random_graph(self, vertices: int) -> npt.NDArray:
        edges = round(self.density * (vertices**2 - vertices) / 2)
        return nx.to_numpy_array(nx.gnm_random_graph(n=vertices, m=edges))

    def name(self) -> str:
        return f"Gnm_{self.density:f}"


@dataclass
class RandomRegularGraph(GraphType):
    degree: int

    def make_random_graph(self, vertices: int) -> npt.NDArray:
        return nx.to_numpy_array(
            nx.random_regular_graph(d=min(self.degree, vertices), n=vertices)
        )

    def name(self) -> str:
        return f"RRG_{self.degree:f}"


# **********************************
# OPTIMIZATION PROBLEMS
# **********************************


def all_bit_strings(N):
    """Return a matrix of shape (2**N, N) of all bit strings that
    can be constructed using 'N' bits. Each row is a different
    configuration, corresponding to the integers 0, 1, 2 up to (2**N)-1"""
    confs = np.arange(2**N, dtype=np.int32)
    return np.array([(confs >> i) & 1 for i in range(N)], dtype=np.uint32)


class ProblemKind(Enum):
    SK = 0  # Sherrington-Kirkpatrick is an alias for Max-Cut
    MAXCUT = 0
    QUBO = 1
    Ising = 2


@dataclass
class Problem:
    kind: ProblemKind
    J: npt.NDArray
    h: Union[npt.NDArray, float] = 0.0

    def energy_vector(self, problem_type=None, normalize=True):
        if problem_type is None:
            problem_type = self.kind
        elif problem_type == "MAXCUT":
            problem_type = ProblemKind.MAXCUT
        elif problem_type == "QUBO":
            problem_type = ProblemKind.QUBO
        elif problem_type == "Ising":
            problem_type = ProblemKind.Ising
        if problem_type == ProblemKind.MAXCUT:
            E = self.MaxCut_energy(self.J)
        elif problem_type == ProblemKind.Ising:
            E = self.Ising_energy(self.J, self.h)
        elif problem_type == ProblemKind.QUBO:
            E = self.QUBO_energy(self.J)
        else:
            raise Exception(f"Uknown problem type {problem_type}")
        if normalize:
            E -= np.min(E)
            E /= np.max(E)
        return E

    @property
    def size(self):
        return len(self.J)

    # We face two energy functionals that have the same purpose. In the QUBO
    # formalism, we write
    #
    #   E[x] = \sum_{ij} 2 x[i] Q[i,j] x[j]
    #
    # In the Ising formalism we write
    #
    #   E[s] = \sum_{ij} (1/2) s[i] J[i,j] s[j] + \sum_i h[i] s[i]
    #
    # Since the bits x[i] ϵ {0,1} are related to the spins s[i] ϵ {-1,1} by
    #
    #   s = 2*x-1   or  x = (s + 1)/2
    #
    # We can write
    #   E[s] = \sum_{ij} (s[i] + 1)Q[i,j](s[j]+1)/2
    #        = \sum_{ij} (1/2) s[i] Q[i,j] s[j] + \sum_{ij} (Q[i,j]+Q[j,i]) s[j]/2
    #        +  \sum_{ij} Q[i,j] / 2
    #
    # This gives a relation between Q and {J,h}
    #   J[i,j] = Q[i,j],
    #   h[j] = \sum_i (Q[i,j] + Q[j,i]) / 2
    @staticmethod
    def QUBO_energy(Q):
        bits = all_bit_strings(len(Q))
        return 2 * ((Q @ bits) * bits).sum(0)

    # The MAXCUT problem is slightly different, as it is defined by
    #
    #   E[x] = \sum_{ij} -2 x[i] Q[i,j] (1 - x[j])
    #
    # When mapping to the Ising model, this means
    #
    #   E[s] = \sum_{ij} -(s[i] + 1)Q[i,j](1 - s[j])/2
    #        = \sum_{ij} (1/2) s[i] Q[i,j] s[j] + \sum_{ij} (Q[i,j]-Q[j,i]) s[j]/2
    #        +  \sum_{ij} Q[i,j] / 2
    #
    # Since the matrix Q is symmetric, we directly obtain
    #   J = Q
    #   h = 0
    @staticmethod
    def MaxCut_energy(Q):
        bits = all_bit_strings(len(Q))
        return -2 * ((Q @ bits) * (1 - bits)).sum(0)

    @staticmethod
    def Ising_energy(J, h):
        """Return the energies of all basis states for an Ising problem
        defined by the Ising matrix J and the magnetic field."""
        spins = 2 * all_bit_strings(len(J)).astype(int) - 1
        return 0.5 * ((J @ spins) * spins).sum(0) + h @ spins

    @staticmethod
    def random_problem(N: int, kind: ProblemKind, graph: GraphType):
        W = graph.make_random_graph(N)
        J = np.array(W) * np.random.normal(loc=0, scale=1, size=(N, N))
        J = np.triu(J, k=1) + np.triu(J, k=1).T
        h = np.random.normal(loc=0, scale=1, size=N)
        return Problem(kind=kind, J=J, h=h)


#
# SAVE AND LOAD FROM HDF5 FILES
#
def save_problems_to_hdf5(filename: str, problems: list[Problem]) -> None:
    def save_one_problem(group: h5py.Group, problem: Problem) -> None:
        group.create_dataset("kind", shape=(1,), data=[problem.kind.value])
        group.create_dataset("J", shape=problem.J.shape, data=problem.J)
        group.create_dataset("h", shape=problem.h.shape, data=problem.h)

    with h5py.File(filename, "w") as file:
        group = file.create_group("problems")
        for i, p in enumerate(problems):
            save_one_problem(group.create_group(f"problem{i:07x}"), p)


def load_problems_from_hdf5(
    filename: str, start: Optional[int] = None, end: Optional[int] = None
) -> list[Problem]:
    def read_one_problem(group: h5py.Group) -> Problem:
        return Problem(
            kind=ProblemKind(group["kind"][0]),
            J=group["J"][:],
            h=group["h"][:],
        )

    def select(groups: list[h5py.Group], start, end):
        if start is not None:
            start = f"{start:07x}"
            end = f"Z" if end is None else f"{end:07x}"
            selected = [
                g for g in groups if (start <= g.name[-7:] and g.name[-7:] < end)
            ]
            print([g.name for g in selected])
            return selected
            exit(0)
        return groups

    with h5py.File(filename, "r") as file:
        group = file["problems"]
        dataset = [
            read_one_problem(problem) for problem in select(group.values(), start, end)
        ]
    return dataset
