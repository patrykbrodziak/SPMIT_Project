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

    def __init__(
        self,
        population_size: int,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.6,
        elitism_rate: float = 0.1,
        extra_initialization_rate: float = 5.0,
        crossover: str = "ox",
        mutation: str = "inv",
        selection: str = "n",
        crossover_schedule_type: str = "const",
        mutation_schedule_type: str = "const",
    ):
        """Initializer"""
        self.distance_matrix = None
        self.history = {"min_fitness": None, "mean_fitness": None, "max_fitness": None, "epoch": None}
        # set operators
        self.mutation = self.mutation_operator[mutation]
        self.crossover = self.crossover_operator[crossover]
        self.selection = self.selection_operator[selection]
        # set operator rate
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.extra_initialization_rate = extra_initialization_rate
        # set schedules
        self.mutation_schedule_type = self.operator_schedules[mutation_schedule_type]
        self.crossover_schedule_type = self.operator_schedules[crossover_schedule_type]

    @staticmethod
    def validate_population(population: tf.Tensor) -> None:
        """Check if all elements in generation array are permutation of input data"""
        if not np.all(
            np.apply_along_axis(lambda individual: np.unique(individual).shape[0] == individual.shape[0], 1, population)
        ):
            raise ValueError("Genetic Operator resulted in invalid representation!")

    @staticmethod
    def initialize_population(
        individual_size: int, population_size: int, extra_initialization_rate: float = 1.0
    ) -> tf.Tensor:
        """
        Initializes population with stochastic method, where
        each individual is randomly permuted order of coordinates

        :param individual_size: number of locations for TSP problem
        :param population_size: number of initial solutions
        :param extra_initialization_rate: number of extra generated solutions, which are discarded if they have fitness
                                          worse than number of solutions given by population size

        :return: array with shape (population_size, num_points) where each element
                 is individual solution
        """
        initial_size = int(population_size * extra_initialization_rate)
        population = tf.broadcast_to(tf.range(0, individual_size), shape=[initial_size, individual_size])
        population = tf.map_fn(np.random.permutation, population)

        return population

    def create_offspring(self, mating_pool: tf.Tensor, num_offspring: int) -> tf.Tensor:
        """
        Create crossover solutions from given mating pool

        :param mating_pool: solutions to choose from
        :param num_offspring: number of generated offspring

        :return: array of offspring solutions
        """
        candidates = tf.random.uniform([num_offspring, 2], maxval=mating_pool.shape[0], dtype="int32")
        parents = tf.gather(mating_pool, candidates)

        return tf.map_fn(self.crossover, parents)

    def mutate(self, mutation_pool: tf.Tensor) -> tf.Tensor:
        """
        Mutate selected solutions by given operator

        :param mutation_pool: solutions to mutate

        :return: array with mutated offspring
        """
        return tf.map_fn(self.mutation, mutation_pool)
