from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from tqdm import tqdm as progress_bar

from src.solvers.genetic import BaseCombinatorialGeneticOptimizer
from src.solvers.utils import slice_update


class TSPGeneticOptimizer(BaseCombinatorialGeneticOptimizer):
    """
    Class holds implementation of genetic optimization
    algorithm for traveling salesman problem
    """

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
        crossover_schedule_type: str = "constant",
        mutation_schedule_type: str = "constant",
    ):
        """Initializer"""
        super().__init__()
        # set operators
        self.mutation = self.mutation_operator[mutation]
        self.crossover_op = self.crossover_op[crossover]
        self.selection = self.selection_operator[selection]
        # set operator rate
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.extra_initialization_rate = extra_initialization_rate

        self.mutation_schedule = mutation_schedule_type
        self.crossover_schedule = crossover_schedule_type

    def fitness(self, specimen: np.array) -> np.array:
        """Calculates fitness for given ordering of coordinates in an array"""
        return np.sum(self.distance_matrix[specimen[1:], specimen[:-1]])

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

    def crossover(self, mating_pool: tf.Tensor, num_offspring: int) -> tf.Tensor:
        """
        Create crossover solutions from given mating pool

        :param mating_pool: solutions to choose from
        :param num_offspring: number of generated offspring

        :return: array of offspring solutions
        """
        candidates = tf.random.uniform([num_offspring, 2], maxval=mating_pool.shape[0], dtype="int32")
        parents = tf.gather(mating_pool, candidates)

        return tf.map_fn(self.crossover_op, parents)

    def mutate(self, mutation_pool: tf.Tensor) -> tf.Tensor:
        """
        Mutate selected solutions by given operator

        :param mutation_pool: solutions to mutate

        :return: array with mutated offspring
        """
        return tf.map_fn(self.mutation, mutation_pool)

    @staticmethod
    def schedules(num_steps: int, rate: float) -> np.array:
        """
        Get dict of mutation schedules as numpy arrays

        :param num_steps: number of iterations in minimize method
        :param rate: crossover or mutation rate

        :return: dict with schedules to choose from
        """
        time_steps = np.linspace(0.001, 1.0, num_steps)
        schedules = {
            "constant": np.full(num_steps, rate),  # constant rate
            # linear functions starting at 0 and ending at selected rate
            "increasing_linear": np.apply_along_axis(lambda step: rate * step, 0, time_steps),
            "decreasing_linear": np.apply_along_axis(lambda step: rate * (1.0 - step), 0, time_steps),
            # square root function
            "increasing_root": np.apply_along_axis(lambda step: rate * np.sqrt(step), 0, time_steps),
            "decreasing_root": np.apply_along_axis(lambda step: rate * np.sqrt(1.0 - step), 0, time_steps),
            # second power functions
            "increasing_square": np.apply_along_axis(lambda step: rate * step ** 2, 0, time_steps),
            "decreasing_square": np.apply_along_axis(lambda step: rate * ((1.0 - step) ** 2), 0, time_steps),
        }

        return schedules

    def minimize(
        self,
        coordinates: np.array,
        num_steps: int,
        patience: int = 50,
        silent: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Minimize TSP for given coordinate points

        :param coordinates: coordinates of cities to minimize TSP
        :param num_steps: number of iterations
        :param patience: number of epochs without improving solution before terminating
        :param silent: if False print progress bar during execution

        :return: tuple with best route and its length
        """
        self.distance_matrix = distance.cdist(coordinates, coordinates)

        crossover_schedule = TSPGeneticOptimizer.schedules(num_steps, self.crossover_rate)[self.crossover_schedule]
        mutation_schedule = TSPGeneticOptimizer.schedules(num_steps, self.mutation_rate)[self.mutation_schedule]

        population = self.initialize_population(coordinates.shape[0], self.population_size, self.extra_initialization_rate)
        self.history["min_fitness"] = np.zeros(num_steps)
        self.history["mean_fitness"] = np.zeros(num_steps)
        self.history["max_fitness"] = np.zeros(num_steps)

        for generation in progress_bar(range(num_steps), disable=silent):
            elite = self.selection(self.fitness, population, int(self.elitism_rate * self.population_size))

            self.validate_population(population.numpy())

            num_to_crossover = int(self.crossover_rate * self.population_size)
            mating_pool = self.selection(self.fitness, population, num_to_crossover)
            offspring = self.crossover(mating_pool, int((1 - self.elitism_rate) * self.population_size))

            self.validate_population(population.numpy())

            num_to_mutate = int(self.mutation_rate * self.population_size)
            to_mutate = tf.random.uniform(
                [num_to_mutate, ], maxval=int((1 - self.elitism_rate) * self.population_size), dtype="int32"
            )
            offspring = slice_update(offspring, indices=to_mutate, updates=self.mutate(tf.gather(offspring, to_mutate)))

            self.validate_population(population.numpy())

            # concatenate all solutions and create next generation
            population = tf.concat([elite, offspring], axis=0)

            fitness = tf.map_fn(self.fitness, population)

            self.history["mean_fitness"][generation] = fitness.numpy().mean()
            self.history["min_fitness"][generation] = fitness.numpy().min()
            self.history["max_fitness"][generation] = fitness.numpy().max()
            self.history["epoch"] = generation

            self.crossover_rate = crossover_schedule[generation]
            self.mutation_rate = mutation_schedule[generation]

            # break condition
            validation = self.history["min_fitness"][generation - patience: generation]
            if np.all(np.diff(validation) == 0) and generation >= patience:
                return population[fitness.numpy().argmin()], fitness.numpy().min()

        return population[fitness.numpy().argmin()], fitness.numpy().min()
