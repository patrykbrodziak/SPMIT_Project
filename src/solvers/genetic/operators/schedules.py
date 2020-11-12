import numpy as np


def constant(num_steps: int, rate: float) -> np.array:
    return np.full(num_steps, rate)


def increasing_linear(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * step, 0, time_steps)


def decreasing_linear(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * (1.0 - step), 0, time_steps)


def increasing_root(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * np.sqrt(step), 0, time_steps)


def decreasing_root(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * np.sqrt(1.0 - step), 0, time_steps)


def increasing_square(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * step ** 2, 0, time_steps)


def decreasing_square(num_steps: int, rate: float) -> np.array:
    time_steps = np.linspace(0.001, 1.0, num_steps)
    return np.apply_along_axis(lambda step: rate * ((1.0 - step) ** 2), 0, time_steps)