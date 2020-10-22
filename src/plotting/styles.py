import seaborn as sns
from matplotlib import pyplot as plt


def set_style() -> None:
    """Set default style to seaborn and matplotlib plots"""
    sns.set()

    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["font.size"] = 12

    return
