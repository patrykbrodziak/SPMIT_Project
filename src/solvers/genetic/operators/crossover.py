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


def cx():
    ...
