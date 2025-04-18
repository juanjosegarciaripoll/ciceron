# **********************************
# EXPERIMENT TO FIND OPTIMAL ANGLES OF MULTI-LAYER QAOA
# **********************************

import numpy as np
import pickle
import time
import argparse
from . import optimization, mympi
from .problems import Problem, ProblemKind, load_problems_from_hdf5
import os
import dataclasses
from typing import Optional
import portalocker
import h5py


@dataclasses.dataclass
class Experiment:
    N: int
    depth: int
    problem_type: str
    kind: ProblemKind
    rho: float = 1.0
    batch_size: int = 100
    batch_number: int = 0
    use_cython: bool = False
    output_file: str = "expts/qaoa.hdf5"
    reset_output_file: bool = False
    smooth_interpolation: bool = False

    def start(self):
        return self.batch_size * self.batch_number

    def end(self):
        return self.batch_size * (self.batch_number + 1)

    def hdf5_group(self, layer: Optional[int] = None) -> str:
        group = f"problem_type={self.problem_type}/rho={self.rho:1.1f}/N={self.N:d}/"
        if self.batch_size == 1:
            group += f"/i={self.start()}"
        if layer:
            group += f"/layer={layer:02d}"
        return group

    def replace(self, *args, **kwdargs):
        return dataclasses.replace(self, *args, **kwdargs)


def obtain_arguments() -> Experiment:
    """Obtain the arguments to run the experiment"""
    parser = argparse.ArgumentParser(
        prog="qaoa_experiment",
        description="Run a batch of experiments",
    )
    #
    # Arguments to the program. The long names (preceded by --) of the options
    # match the names of the fields in our Experiment dataclass
    #
    parser.add_argument("-N", type=int, required=True, help="number of qubits")
    parser.add_argument(
        "-d", "--depth", type=int, required=True, help="number of layers"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=100,
        required=False,
        help="number of experiments per run",
    )
    parser.add_argument(
        "-n",
        "--batch-number",
        type=int,
        default=0,
        required=False,
        help="which batch to run",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        required=False,
        help="density of graph for this problem",
    )
    parser.add_argument(
        "-t", "--problem_type", default="MAXCUT", help="problem type: QUBO, MAXCUT"
    )
    parser.add_argument(
        "--smooth",
        action="count",
        default=0,
        help="smoothly interpolate initial conditions for subsequent layers",
    )
    parser.add_argument(
        "--use-cython",
        action="count",
        default=0,
        help="use Cython version of the library",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="output file name",
    )
    parser.add_argument(
        "--clean",
        action="count",
        default=0,
        help="delete output file before starting jobs",
    )
    args = parser.parse_args()
    if args.output is None:
        if args.use_cython > 0:
            args.output = "expts/qaoa_cython.hdf5"
        else:
            args.output = "expts/qaoa.hdf5"
    if args.problem_type == "MAXCUT":
        kind = ProblemKind.MAXCUT
    elif args.problem_type == "QUBO":
        kind = ProblemKind.QUBO
    else:
        raise Exception(f"Unknown problem type {args.problem_type}")
    if args.N < 1:
        raise Exception(f"Invalid number of qubits {args.N}")
    if args.depth < 1 or args.depth > 40:
        raise Exception(f"Invalid circuit depth {args.depth}")
    return Experiment(
        N=args.N,
        depth=args.depth,
        problem_type=args.problem_type,
        kind=kind,
        rho=args.rho,
        batch_size=args.batch_size,
        batch_number=args.batch_number,
        use_cython=(args.use_cython > 0),
        reset_output_file=(args.clean > 0),
        smooth_interpolation=(args.smooth > 0),
    )


def load_experiment(expt: Experiment) -> list[Problem]:
    """Load all problems associated to one experiment."""
    problems_file = f"problems/Prob_QUBO_Nq{expt.N}_Nly_iter1000_rho1.0.hdf5"
    start = expt.batch_number * expt.batch_size
    end = start + expt.batch_size
    return load_problems_from_hdf5(problems_file, start, end)


def extend_experiment(expt: Experiment) -> list[Experiment]:
    """Separate batches into separate problems."""
    return [
        expt.replace(batch_size=1, batch_number=n)
        for n in range(expt.start(), expt.end())
    ]


def reset_hdf5_file(filename: str) -> None:
    lockfile = filename + ".lock"
    with portalocker.Lock(lockfile, "wb") as file:
        if os.path.exists(filename):
            os.unlink(filename)


def save_experiment_result(expt: Experiment, data: dict) -> None:
    lockfile = expt.output_file + ".lock"
    with portalocker.Lock(lockfile, "wb") as flock:
        with h5py.File(expt.output_file, "a") as file:
            group = file.require_group(expt.hdf5_group(data["Depth"]))
            print(f"Saving group {group} to {expt.output_file}")
            for key, value in data.items():
                if value is None:
                    value = []
                array_value = np.asarray(value)
                group.require_dataset(
                    key,
                    shape=array_value.shape,
                    dtype=array_value.dtype,
                    data=array_value,
                )


def load_hdf5_results_as_dataframe(filename: str):
    import pandas

    table, fields = load_hdf5_results(filename)
    columns: list[str] = [""] * len(fields)
    for key, ndx in fields.items():
        columns[ndx] = key
    return pandas.DataFrame(table, columns=columns)


def load_hdf5_results(filename: str) -> list[dict]:
    def read_group(group, output, fields):
        first = True
        dataset = False
        for key, value in group.items():
            if first:
                if isinstance(value, h5py.Dataset):
                    row = [None] * len(fields)
                    row[0] = group.name
                    dataset = True
                else:
                    dataset = False
                first = False
            if dataset:
                n = fields.get(key, None)
                if n is None:
                    fields[key] = len(fields)
                    row.append(value[()])
                else:
                    row[n] = value[()]
            else:
                read_group(value, output, fields)
        if dataset:
            output.append(row)
        return output

    fields = {"path": 0}
    with h5py.File(filename, "r") as file:
        output = read_group(file, [], fields)
    return output, fields


def run_experiment(expt: Experiment):
    starttime_global = time.time()
    problems = load_experiment(expt)
    if len(problems) == 0:
        print(f"Missing problem {expt.hdf5_group()}")
        return
    assert len(problems) == 1
    problem = problems[0]
    E = problem.energy_vector(problem_type=expt.kind)  # Load the instance
    degeneracy = optimization.GS_degeneracy(E)
    previous = None
    for layers in range(1, expt.depth + 1):
        starttime_local = time.time()
        previous = optimization.OptimalAnglesExperiment(
            E,
            layers=layers,
            previous=previous,
            fit=True,
            use_cython=expt.use_cython,
            smooth_interpolation=expt.smooth_interpolation,
        )
        finishtime_local = time.time() - starttime_local
        print(
            f"Energy = {previous.energy},\ntheta = {previous.theta},\ngamma = {previous.gamma}\ntime = {finishtime_local}"
        )
        data = {
            "Kind": expt.kind.value,
            "Size": expt.N,
            "Depth": layers,
            "Theta": previous.theta,
            "Gamma": previous.gamma,
            "Energy": previous.energy,
            "Energy std": previous.E_std,
            "Participation energy": previous.E_PR,
            "GS probability": previous.pgs,
            "Beta Total": previous.beta,
            "Beta High": previous.βlin_1,
            "Beta Low": previous.βlin_2,
            "Number Participation states": len(previous.psi_1),
            "Intermediate Times": finishtime_local,
            "Optimizer iterations": previous.nit,
            "Total time": 0,
            "Degeneracy": degeneracy,
        }
        save_experiment_result(expt, data)

    finishtime_global = time.time() - starttime_global
    print(
        f"Ran job {expt.start()} with {expt.N} qubits in {finishtime_global} seconds -- Depth {expt.depth}"
    )
    return []


def main_job(
    expt: Experiment,
    mpi=False,
    workers=1,
    outputfile=None,
):
    os.makedirs(os.path.dirname(expt.output_file), exist_ok=True)
    if expt.reset_output_file and os.path.exists(expt.output_file):
        reset_hdf5_file(expt.output_file)

    work_to_do = extend_experiment(expt)
    print(f"Launching workers for {outputfile}")
    if mpi:
        mympi.mpi_split_job(
            run_experiment, work_to_do, workers=workers, root_works=True
        )
    else:
        mympi.workers_split_job(workers, run_experiment, work_to_do, root_works=False)
