import tensorflow as tf


def binary_tournament():
    ...


def n_fittest(fitness, population: tf.Tensor, n: int) -> tf.Tensor:
    """Choose N individuals with highest fitness"""
    fitness = tf.map_fn(fitness, population, dtype="float32")
    return tf.gather(population, tf.argsort(fitness)[:n])


def tournament():
    ...
