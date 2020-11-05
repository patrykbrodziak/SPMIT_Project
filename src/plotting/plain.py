import numpy as np
from matplotlib import pyplot as plt


def history(history: dict) -> None:
    """Plots optimization history"""
    final_epoch = history["epoch"]
    num_steps = history["mean_fitness"].shape[0]
    time_steps = np.linspace(0, num_steps, num_steps)

    plt.plot(time_steps[:final_epoch], history["mean_fitness"][:final_epoch])
    plt.plot(time_steps[:final_epoch], history["min_fitness"][:final_epoch])
    plt.plot(time_steps[:final_epoch], history["max_fitness"][:final_epoch])

    plt.legend(["Mean Fitness", "Min Fitness", "Max Fitness"])

    return


def route(
    order: np.array,
    points: np.array,
    point_color: str = "r",
    arrow_color: str = "b",
    point_size: int = 80,
) -> None:
    """
    Plots given route and data points

    :param order: order of traversed points, 1D integer array with ordering of points
    :param points: coordinates in plane of data points, 2D numeric array with coordinates of all points
    :param point_color: color points are marked with
    :param arrow_color: color arrows are marked with
    :param point_size: size of point markers
    """
    if len(order) != len(points):
        raise ValueError("Array lengths must be equal!")

    axes = plt.axes()

    for index in range(1, len(points)):
        x = points[int(order[index - 1])]
        y = points[int(order[index])]
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        axes.arrow(x[0], x[1], dx, dy, color=arrow_color)

    plt.scatter(points[:, 0], points[:, 1], color=point_color, s=point_size)
