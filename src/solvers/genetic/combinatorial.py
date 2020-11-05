import numpy as np
import tensorflow as tf

from src.solvers.genetic import operators


class BaseCombinatorialGeneticOptimizer:
    """
    Base class for Combinatorial problems genetic optimizers
    """
    crossover_operator = {
        "ox": operators.crossover.ox,
        "cx": operators.crossover.cx,
    }

    mutation_operator = {
        "inv": operators.mutation.invert,
        "ins": operators.mutation.insert,
        "ex": operators.mutation.exchange,
        "dis": operators.mutation.displace,
    }

    selection_operator = {
        "bt": operators.selection.binary_tournament,
        "n": operators.selection.n_fittest,
        "t": operators.selection.tournament,
    }

    operator_schedules = {
        "const": operators.schedules.constant,
        "increasing_linear": operators.schedules.increasing_linear,
        "decreasing_linear": operators.schedules.decreasing_linear,
        "increasing_root": operators.schedules.increasing_root,
        "decreasing_root": operators.schedules.decreasing_root,
        "increasing_square": operators.schedules.increasing_square,
        "decreasing_square": operators.schedules.decreasing_square,
    }

    def __init__(self):
        self.distance_matrix = None
        self.history = {"min_fitness": None, "mean_fitness": None, "max_fitness": None, "epoch": None}

    @staticmethod
    def validate_population(population: tf.Tensor) -> None:
        """Check if all elements in generation array are permutation of input data"""
        if not np.all(
            np.apply_along_axis(lambda individual: np.unique(individual).shape[0] == individual.shape[0], 1, population)
        ):
            raise ValueError("Genetic Operator resulted in invalid representation!")
