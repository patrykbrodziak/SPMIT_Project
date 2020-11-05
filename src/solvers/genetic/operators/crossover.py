import numpy as np
import tensorflow as tf

from src.solvers.utils import slice_update


def ox(parents: tf.Tensor) -> tf.Tensor:
    """
    Creates offspring from two parent arrays
    Offspring has a partial tour from both parent arrays

    :param parents: Tensor of two individual solutions to participate in crossover

    :return: offspring array being a combination of mother and father arrays
    """
    mother, father = parents
    offspring = tf.fill([mother.shape[0]], -1)  # initialize offspring with negative values
    # take random slice of mother tensor
    low, high = tf.sort(tf.random.uniform([2, ], maxval=mother.shape[0], dtype="int32"))
    offspring = slice_update(offspring, tf.range(low, high), mother[low:high])  # fill offspring with mother slice
    # gather remaining elements from father tensor
    to_update = tf.where(offspring < 0)
    updates = tf.gather(
        father, tf.where(tf.logical_not(tf.numpy_function(np.isin, [father, offspring], Tout="float32")))
    )
    # fill in offspring with father slice
    offspring = slice_update(offspring, to_update, updates)

    return offspring


def cx(parents: tf.Tensor) -> tf.Tensor:
    """
    Creates offspring from two parent arrays using Cycle Crossover operator

    :param parents: Tensor of two individual solutions to participate in crossover

    :return: offspring array being a combination of mother and father arrays
    """
    mother, father = parents.numpy()
    offspring = np.full((mother.shape[0]), -1)

    index = np.random.randint(mother.shape[0])

    while not np.isin(mother[index], offspring):  # loop until encountered index which is already in offspring
        offspring[index] = mother[index]
        index = np.where(mother == father[index])

    leftover_indices = np.where(offspring < 0)  # pick all empty indices in offspring
    offspring[leftover_indices] = father[leftover_indices]

    return tf.convert_to_tensor(offspring)

