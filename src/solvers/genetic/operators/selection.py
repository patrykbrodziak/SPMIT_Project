import tensorflow as tf


def binary_tournament():
    ...


def n_fittest(self, population: tf.Tensor, n_fittest: int) -> tf.Tensor:
    """Choose N individuals with highest fitness"""
    fitness = tf.map_fn(self.fitness, population, dtype="float32")
    return tf.gather(population, tf.argsort(fitness)[:n_fittest])


def tournament():
    ...
