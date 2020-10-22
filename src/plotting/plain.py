import numpy as np
from matplotlib import pyplot as plt


def plot_route(
        route: np.array,
        points: np.array,
        point_color: str = "r",
        arrow_color: str = "b",
        point_size: int = 80,
) -> None:
    """
    Plots given route and data points

    :param route: order of traversed points, 1D integer array with ordering of points
    :param points: coordinates in plane of data points, 2D numeric array with coordinates of all points
    :param point_color: color points are marked with
    :param arrow_color: color arrows are marked with
    :param point_size: size of point markers
    """
    if len(route) != len(points):
        raise ValueError("Array lengths must be equal!")

    axes = plt.axes()

    for index in range(1, len(points)):
        x = points[int(route[index-1])]
        y = points[int(route[index])]
        dx = y[0] - x[0]
        dy = y[1] - x[1]
        axes.arrow(x[0], x[1], dx, dy, color=arrow_color)

    plt.scatter(points[:, 0], points[:, 1], color=point_color, s=point_size)
