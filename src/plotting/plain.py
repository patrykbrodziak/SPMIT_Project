import numpy as np
import seaborn as sns
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
    route_point_order: np.array,
    scatter_points: np.array,
    point_color: str = "r",
    arrow_color: str = "b",
    point_size: int = 80,
) -> None:
    """
    Plots given route and data points
    :param route_point_order: order of traversed points, 1D integer array with ordering of points
    :param scatter_points: coordinates in plane of data points, 2D numeric array with coordinates of all points
    :param point_color: color points are marked with
    :param arrow_color: color arrows are marked with
    :param point_size: size of point markers
    """
    if len(route_point_order) != len(scatter_points):
        raise ValueError("Array lengths must be equal!")

    axes = plt.axes()

    for index in range(1, len(scatter_points)):
        x = scatter_points[int(route_point_order[index - 1])]
        y = scatter_points[int(route_point_order[index])]

        dx = y[0] - x[0]
        dy = y[1] - x[1]

        axes.arrow(x[0], x[1], dx, dy, color=arrow_color)

    plt.scatter(scatter_points[:, 0], scatter_points[:, 1], color=point_color, s=point_size)


def multiple_routes(
    route_orders: np.array,
    scatter_points: np.array,
    route_args: dict = None,
    scatter_args: dict = None,
):
    """Plots multiple routes on single plot"""
    route_args = route_args or {}
    scatter_args = scatter_args or {}

    colors = sns.color_palette()
    axes = plt.axes()

    for route_index, route_points_order in enumerate(route_orders):
        for point_index in range(1, len(route_points_order)):
            x = scatter_points[int(route_points_order[point_index - 1])]
            y = scatter_points[int(route_points_order[point_index])]

            dx = y[0] - x[0]
            dy = y[1] - x[1]

            axes.arrow(x[0], x[1], dx, dy, color=colors[route_index], **route_args)

    plt.scatter(scatter_points[:, 0], scatter_points[:, 1], **scatter_args)
