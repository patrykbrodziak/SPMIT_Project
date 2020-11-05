import tensorflow as tf

from src.solvers.utils import slice_update


def invert(specimen: tf.Tensor) -> tf.Tensor:
    """
    Creates offspring by inverting random sequence in individual solution

    :param specimen: solution to mutate

    :return: mutated offspring
    """
    low, high = tf.sort(tf.random.uniform([2, ], maxval=specimen.shape[0], dtype="int32"))
    return slice_update(specimen, tf.range(low, high), tf.reverse(specimen[low:high], [0]))


def insert():
    ...


def exchange():
    ...


def displace():
    ...
