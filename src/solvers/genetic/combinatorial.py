from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from tqdm import tqdm as progress_bar

from src.solvers.genetic import operators
from src.solvers.utils import slice_update


class BaseCombinatorialGeneticOptimizer:
    """
    Base class for Combinatorial problems genetic optimizers
    """
    crossover_op = {
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
