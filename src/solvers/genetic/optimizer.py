from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.spatial import distance
from tqdm import tqdm as progress_bar

from .utils import slice_update


class GeneticOptimizer:
    """
    Class holds implementation of genetic optimization
    algorithm for traveling salesman problem
    """
    def __init__(self, coordinates: np.array):
        """Initializer"""
        self.neighbourhood_matrix = distance.cdist(coordinates, coordinates)
        self.history = {'min_fitness': None, 'mean_fitness': None, 'max_fitness': None, 'epoch': None}
        self.crossover_operator = {
            'OX': GeneticOptimizer.create_offspring_ox,
            'CX': GeneticOptimizer.create_offspring_cx
        }
        self.mutation_operator = {
            'Inversion': GeneticOptimizer.mutate_gene_invert,
            'Insertion': GeneticOptimizer.mutate_gene_insert,
            'Displacement': GeneticOptimizer.mutate_gene_displace,
            'Exchange': GeneticOptimizer.mutate_gene_exchange
        }

    @staticmethod
    def vectorized_validate(data_array: tf.Tensor) -> bool:
        """
        Check if all elements in generation
        array are permutation of input data

        :param data_array: data for TSP

        :return: True if data is correct
        """
        return np.all(np.apply_along_axis(
            lambda individual:
                np.unique(individual).shape[0] == individual.shape[0],
            1, data_array))

    def select_n_fittest(self, population: tf.Tensor, num_fittest: int) -> tf.Tensor:
        """
        Choose N individuals with highest fitness
        """
        fitness = tf.map_fn(self.get_fitness, population, dtype="float32")
        return tf.gather(population, tf.argsort(fitness)[:num_fittest])

    def get_fitness(self, specimen: np.array) -> np.array:
        """Calculates fitness for given ordering of coordinates in an array"""
        return np.sum(self.neighbourhood_matrix[specimen[1:], specimen[:-1]])

    def initialize_population(
            self, individual_size: int, population_size: int, extra_initialization_rate: float = 1.0
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

    @staticmethod
    def create_offspring_ox(parents: tf.Tensor) -> tf.Tensor:
        """
        Creates offspring from two parent arrays
        Offspring has a partial tour from both parent arrays

        :param parents: Tensor of two individual solutions to participate in crossover

        :return: offspring array being a combination of mother and father arrays
        """
        mother, father = parents
        offspring = tf.fill([mother.shape[0]], -1)  # initialize offspring with negative values
        low, high = tf.sort(tf.random.uniform([2, ], maxval=mother.shape[0], dtype="int32"))
        offspring = slice_update(offspring, tf.range(low, high), mother[low:high])
        to_update = tf.where(offspring < 0)
        updates = tf.gather(father,
                            tf.where(tf.logical_not(tf.numpy_function(np.isin, [father, offspring], Tout="float32"))))

        return slice_update(offspring, to_update, updates)

    # TODO: Fill in
    @staticmethod
    def create_offspring_cx(parents: tf.Tensor) -> tf.Tensor:
        """
        Creates offspring from two parent arrays, preserving
        as much sequence information from first parent as possible
        and completing information with genome from second parent

        :param parents: Tensor of two individual solutions to participate in crossover

        :return: offspring array being a combination of mother and father arrays
        """
        ...

    def crossover(self, mating_pool: tf.Tensor, num_offspring: int, operator: str) -> tf.Tensor:
        """
        Create crossover solutions from given mating pool

        :param mating_pool: solutions to choose from
        :param num_offspring: number of generated offspring
        :param operator: which crossover operator to use

        :return: array of offspring solutions
        """
        candidates = tf.random.uniform([num_offspring, 2], maxval=mating_pool.shape[0], dtype="int32")
        parents = tf.gather(mating_pool, candidates)
        return tf.map_fn(self.crossover_operator[operator], parents)

    @staticmethod
    def mutate_gene_invert(specimen: tf.Tensor) -> tf.Tensor:
        """
        Creates offspring by inverting random sequence in individual solution

        :param specimen: solution to mutate

        :return: mutated offspring
        """
        low, high = tf.sort(tf.random.uniform([2, ], maxval=specimen.shape[0], dtype="int32"))
        return slice_update(specimen, tf.range(low, high), tf.reverse(specimen[low:high], [0]))

    @staticmethod
    def mutate_gene_insert(specimen: tf.Tensor) -> tf.Tensor:
        """
        Creates mutated offspring by selecting random gene(index in an array)
        and inserting it into random place in the same array

        :param specimen: solution to mutate

        :return: mutated offspring
        """
        ...

    @staticmethod
    def mutate_gene_displace(specimen: tf.Tensor) -> tf.Tensor:
        """
        Creates mutated offspring by selecting random sequence
        and inserting it into different place in an array

        :param specimen: solution to mutate

        :return: mutated offspring
        """
        ...

    @staticmethod
    def mutate_gene_exchange(specimen: tf.Tensor) -> tf.Tensor:
        """
        Creates mutated offspring by swapping two randomly selected genes

        :param specimen: solution to mutate

        :return: mutated offspring
        """
        ...

    def mutate(self, mutation_pool: tf.Tensor, operator: str) -> tf.Tensor:
        """
        Mutate selected solutions by given operator

        :param mutation_pool: solutions to mutate
        :param operator: mutation operator

        :return: array with mutated offspring
        """
        return tf.map_fn(self.mutation_operator[operator], mutation_pool)

    @staticmethod
    def schedules(num_steps: int, rate: float) -> np.array:
        """
        Get dict of mutation schedules as numpy arrays

        :param num_steps: number of iterations in minimize method
        :param rate: crossover or mutation rate

        :return: dict with schedules to choose from
        """
        time_steps = np.linspace(0.001, 1.0, num_steps)
        schedules = {'constant': np.full(num_steps, rate),  # constant rate
                     # linear functions starting at 0 and ending at selected rate
                     'increasing_linear': np.apply_along_axis(lambda step: rate*step, 0, time_steps),
                     'decreasing_linear': np.apply_along_axis(lambda step: rate*(1.0-step), 0, time_steps),
                     # square root function
                     'increasing_root': np.apply_along_axis(lambda step: rate*np.sqrt(step), 0, time_steps),
                     'decreasing_root': np.apply_along_axis(lambda step: rate*np.sqrt(1.0 - step), 0, time_steps),
                     # second power functions
                     'increasing_square': np.apply_along_axis(lambda step: rate*step ** 2, 0, time_steps),
                     'decreasing_square': np.apply_along_axis(lambda step: rate*((1.0 - step) ** 2), 0, time_steps)}

        return schedules

    def minimize(self,
                 data: np.array,
                 num_steps: int,
                 population_size: int,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.6,
                 elitism_rate: float = 0.1,
                 extra_initialization_rate: float = 5.0,
                 crossover_operator: str = 'OX',
                 mutation_operator: str = 'Exchange',
                 crossover_schedule_type: str = 'constant',
                 mutation_schedule_type: str = 'constant',
                 patience: int = 50,
                 logging: bool = False,
                 logging_path: str = None,
                 ) -> Tuple[Any, Any]:
        """
        Minimize TSP for given coordinate points

        :param data: coordinates of cities to minimize TSP
        :param num_steps: number of iterations
        :param population_size: number of individuals in a generation
        :param crossover_rate: fraction of parents for population
        :param mutation_rate: fraction of mutated solutions for each epoch
        :param elitism_rate: fraction of copied solutions
        :param extra_initialization_rate: number of over initialized solutions
        :param crossover_operator: which type of crossover to use
        :param mutation_operator: which type of mutation to use
        :param crossover_schedule_type: crossover schedule
        :param mutation_schedule_type: mutation schedule
        :param patience: number of epochs without improving solution before terminating
        :param logging: if True saves best route after each generation
        :param logging_path: path to directory where logs will be saved

        :return: tuple with best route and its length
        """
        population = self.initialize_population(data.shape[0], population_size, extra_initialization_rate)
        self.history['min_fitness'] = np.zeros(num_steps)
        self.history['mean_fitness'] = np.zeros(num_steps)
        self.history['max_fitness'] = np.zeros(num_steps)

        crossover_schedule = GeneticOptimizer.schedules(num_steps, crossover_rate)[crossover_schedule_type]
        mutation_schedule = GeneticOptimizer.schedules(num_steps, mutation_rate)[mutation_schedule_type]

        for generation in progress_bar(range(num_steps)):
            elite = self.select_n_fittest(population, int(elitism_rate * population_size))

            if not self.vectorized_validate(population.numpy()):
                raise ValueError('Error in generation creation')

            num_to_crossover = int(crossover_rate * population_size)
            mating_pool = self.select_n_fittest(population, num_to_crossover)
            offspring = self.crossover(mating_pool, int((1 - elitism_rate) * population_size), crossover_operator)

            if not self.vectorized_validate(population.numpy()):
                raise ValueError('Error in crossover functions')

            num_to_mutate = int(mutation_rate * population_size)
            to_mutate = tf.random.uniform([num_to_mutate, ], maxval=int((1 - elitism_rate) * population_size), dtype="int32")
            offspring = slice_update(offspring, indices=to_mutate, updates=self.mutate(tf.gather(offspring, to_mutate), mutation_operator))

            if not self.vectorized_validate(population.numpy()):
                raise ValueError('Error in mutation functions')

            # concatenate all solutions and create next generation
            population = tf.concat([elite, offspring], axis=0)

            fitness = tf.map_fn(self.get_fitness, population)

            self.history['mean_fitness'][generation] = fitness.numpy().mean()
            self.history['min_fitness'][generation] = fitness.numpy().min()
            self.history['max_fitness'][generation] = fitness.numpy().max()
            self.history['epoch'] = generation

            crossover_rate = crossover_schedule[generation]
            mutation_rate = mutation_schedule[generation]

            if logging:
                np.savetxt(logging_path, population[fitness.numpy().argmin()])

            # break condition
            validation = self.history['min_fitness'][generation - patience:generation]
            if np.all(np.diff(validation) == 0) and generation >= patience:
                return population[fitness.numpy().argmin()], fitness.numpy().min()

        return population[fitness.numpy().argmin()], fitness.numpy().min()

    def plot_history(self) -> None:
        """
        Plots optimization history
        """
        final_epoch = self.history['epoch']
        num_steps = self.history['mean_fitness'].shape[0]
        time_steps = np.linspace(0, num_steps, num_steps)

        plt.plot(time_steps[:final_epoch], self.history['mean_fitness'][:final_epoch])
        plt.plot(time_steps[:final_epoch], self.history['min_fitness'][:final_epoch])
        plt.plot(time_steps[:final_epoch], self.history['max_fitness'][:final_epoch])
        plt.grid()
