import numpy as np
import tensorflow as tf

from src.solvers.utils import slice_update


def displace(specimen: tf.Tensor) -> tf.Tensor:
    """
    Creates mutated offspring by selecting random gene(index in an array)
    and inserting it into random place in the same array

    :param specimen: solution to mutate

    :return: mutated offspring
    """
    specimen = specimen.numpy()

    gene_id, placement = np.random.randint(0, len(specimen), 2)
    gene = specimen[gene_id]

    specimen = np.delete(specimen, gene_id)
    specimen = np.insert(specimen, placement, gene)

    return tf.convert_to_tensor(specimen)


def exchange(specimen: tf.Tensor) -> tf.Tensor:
    """
    Creates mutated offspring by swapping two randomly selected genes

    :param specimen: solution to mutate

    :return: mutated offspring
    """
    specimen = specimen.numpy()

    low, high = np.random.randint(0, len(specimen), 2)
    specimen[[low, high]] = specimen[[high, low]]

    return tf.convert_to_tensor(specimen)


def insert(specimen: tf.Tensor) -> tf.Tensor:
    """
    Creates mutated offspring by selecting random gene(index in an array)
    and inserting it into random place in the same array

    :param specimen: solution to mutate

    :return: mutated offspring
    """
    specimen = specimen.numpy()

    gene_id, placement = np.random.randint(0, len(specimen), 2)
    gene = specimen[gene_id]

    specimen = np.delete(specimen, gene_id)
    specimen = np.insert(specimen, placement, gene)

    return tf.convert_to_tensor(specimen)


def invert(specimen: tf.Tensor) -> tf.Tensor:
    """
    Creates offspring by inverting random sequence in individual solution

    :param specimen: solution to mutate

    :return: mutated offspring
    """
    low, high = tf.sort(tf.random.uniform([2, ], maxval=specimen.shape[0], dtype="int32"))
    return slice_update(specimen, tf.range(low, high), tf.reverse(specimen[low:high], [0]))
