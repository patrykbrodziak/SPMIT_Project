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
        crossover_schedule_type: str = "const",
        mutation_schedule_type: str = "const",
    ):
        """Initializer"""
        super().__init__(
            population_size,
            crossover_rate,
            mutation_rate,
            elitism_rate,
            extra_initialization_rate,
            crossover,
            mutation,
            selection,
            crossover_schedule_type,
            mutation_schedule_type,
        )

    def fitness(self, specimen: np.array) -> np.array:
        """Calculates fitness for given ordering of coordinates in an array"""
        return np.sum(self.distance_matrix[specimen[1:], specimen[:-1]])

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

        population = self.initialize_population(coordinates.shape[0], self.population_size, self.extra_initialization_rate)
        self.history["min_fitness"] = np.zeros(num_steps)
        self.history["mean_fitness"] = np.zeros(num_steps)
        self.history["max_fitness"] = np.zeros(num_steps)

        crossover_schedule = self.crossover_schedule_type(num_steps, self.crossover_rate)
        mutation_schedule = self.mutation_schedule_type(num_steps, self.mutation_rate)

        for generation in progress_bar(range(num_steps), disable=silent):
            elite = self.selection(self.fitness, population, int(self.elitism_rate * self.population_size))

            self.validate_population(population.numpy())

            num_to_crossover = int(self.crossover_rate * self.population_size)
            mating_pool = self.selection(self.fitness, population, num_to_crossover)
            offspring = self.create_offspring(mating_pool, int((1 - self.elitism_rate) * self.population_size))

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

            validation = self.history["min_fitness"][generation - patience: generation]
            if np.all(np.diff(validation) == 0) and generation >= patience:
                return population[fitness.numpy().argmin()], fitness.numpy().min()

        return population[fitness.numpy().argmin()], fitness.numpy().min()
